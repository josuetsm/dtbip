import functools
import time
import os

import numpy as np
import scipy.sparse as sparse
from sklearn.decomposition import NMF
import tensorflow as tf
import tensorflow_probability as tfp
import tbip.tbip as tbip

# @title Helper functions for printing topics and ordered ideal points

def get_topics(neutral_mean,
               negative_mean,
               positive_mean,
               vocabulary,
               print_to_terminal=True,
               words_per_topic=10):
    num_topics, num_words = neutral_mean.shape
    top_neutral_words = np.argsort(-neutral_mean, axis=1)
    top_negative_words = np.argsort(-negative_mean, axis=1)
    top_positive_words = np.argsort(-positive_mean, axis=1)
    topic_strings = []
    for topic_idx in range(num_topics):
        neutral_start_string = "Neutral  {}:".format(topic_idx)
        neutral_row = [vocabulary[word] for word in
                       top_neutral_words[topic_idx, :words_per_topic]]
        neutral_row_string = ", ".join(neutral_row)
        neutral_string = " ".join([neutral_start_string, neutral_row_string])

        positive_start_string = "Positive {}:".format(topic_idx)
        positive_row = [vocabulary[word] for word in
                        top_positive_words[topic_idx, :words_per_topic]]
        positive_row_string = ", ".join(positive_row)
        positive_string = " ".join([positive_start_string, positive_row_string])

        negative_start_string = "Negative {}:".format(topic_idx)
        negative_row = [vocabulary[word] for word in
                        top_negative_words[topic_idx, :words_per_topic]]
        negative_row_string = ", ".join(negative_row)
        negative_string = " ".join([negative_start_string, negative_row_string])

        if print_to_terminal:
            topic_strings.append(negative_string)
            topic_strings.append(neutral_string)
            topic_strings.append(positive_string)
            topic_strings.append("==========")
        else:
            topic_strings.append("  \n".join(
                [negative_string, neutral_string, positive_string]))

    if print_to_terminal:
        all_topics = "{}\n".format(np.array(topic_strings))
    else:
        all_topics = np.array(topic_strings)
    return all_topics


def get_ideal_points(ideal_point_loc, author_map, print_to_terminal=True):
    """Print ideal point ordering for Tensorboard."""
    if print_to_terminal:
        offset = 5
        sorted_authors = author_map[np.argsort(ideal_point_loc)]
        authors_by_line = [", ".join(sorted_authors[i * offset:i * offset + offset])
                           for i in range((len(author_map) - 1) // offset + 1)]
        sorted_list = ("Sorted ideal points:"
                       "\n ==================== \n{}"
                       "\n ==================== ".format(",\n".join(authors_by_line)))
    else:
        sorted_list = ", ".join(author_map[np.argsort(ideal_point_loc)])

    return sorted_list

# Función que estima el ELBO
# ELBO es el valor a optimizar en el modelo
# Approximate variational Lognormal ELBO using reparameterization.

def get_elbo(counts,  # A matrix with shape `[batch_size, num_words]`.
             document_indices,  # An int-vector with shape `[batch_size]`.
             author_indices,  # An int-vector with shape `[batch_size]`.
             author_weights,
             # A vector with shape `[num_authors]`, constituting how lengthy the opinion is above average.
             objective_topic_distribution,
             # A positive `Distribution` object with parameter shape `[num_topics, num_words]`.
             document_distribution,
             # A positive `Distribution` object with parameter shape `[num_documents, num_topics]`.
             ideological_topic_distribution,
             # A positive `Distribution` object with parameter shape `[num_topics, num_words]`.
             ideal_point_distribution,  # A `Distribution` object over [0, 1] with parameter_shape `[num_authors]`.
             num_documents,  # The number of documents in the total data set (used to calculate log-likelihood scale).
             batch_size,  # Batch size (used to calculate log-likelihood scale).
             num_samples=1):  # Number of Monte-Carlo samples.
    '''
    Returns:
      elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value is
        averaged across samples and summed across batches.
    '''
    # Get samples (1) from distributions
    document_samples = document_distribution.sample(num_samples)
    objective_topic_samples = objective_topic_distribution.sample(num_samples)
    ideological_topic_samples = ideological_topic_distribution.sample(num_samples)
    ideal_point_samples = ideal_point_distribution.sample(num_samples)

    # Get number of topics

    #### 1. Compute log_prior
    # Get log prios from samples
    document_log_prior = get_log_prior(document_samples, 'gamma')
    objective_topic_log_prior = get_log_prior(objective_topic_samples, 'gamma')
    ideological_topic_log_prior = get_log_prior(ideological_topic_samples, 'normal')
    ideal_point_log_prior = tfp.distributions.Normal(loc=0.0, scale=1.0)
    ideal_point_log_prior = tf.reduce_sum(ideal_point_log_prior.log_prob(ideal_point_samples), axis=1)

    log_prior = (document_log_prior +
                 objective_topic_log_prior +
                 ideological_topic_log_prior +
                 ideal_point_log_prior)

    #### 2. Compute count_log_likelihood

    # Document-topics in batch
    selected_document_samples = tf.gather(document_samples,
                                          document_indices,
                                          axis=1)

    # Ideal points gather by author_indices in batch (autores se repiten pues tienen 1 < documento)
    selected_ideal_points = tf.gather(ideal_point_samples,
                                      author_indices,
                                      axis=1)

    # Compute exp(x_a_d * eta_kv) equation (3)
    selected_ideological_topic_samples = tf.exp(
        selected_ideal_points[:, :, tf.newaxis, tf.newaxis] *  # add 2 new axis to be able to multiply (1, 512, 1, 1)
        ideological_topic_samples[:, tf.newaxis, :, :])

    # Normalize by how lengthy the author's opinion is.
    selected_author_weights = tf.gather(author_weights, author_indices)

    # Compute normalization equation (4)
    selected_ideological_topic_samples = (
            selected_author_weights[tf.newaxis, :, tf.newaxis, tf.newaxis] *
            selected_ideological_topic_samples)

    # Compute rate for Poisson distribution equation (3)
    rate = tf.reduce_sum(
        selected_document_samples[:, :, :, tf.newaxis] *  # theta_dk (batch 512)
        objective_topic_samples[:, tf.newaxis, :, :] *  # beta_kv (no batch) add chanel in 0
        selected_ideological_topic_samples[:, :, :, :],  # exp(x_a_d * eta_kv) (batch 512)
        axis=2)

    # Compute Poisson distribution con parámetros de batch y sample
    count_distribution = tfp.distributions.Poisson(rate=rate)

    # Need to un-sparsify the counts to evaluate log-likelihood.
    count_log_likelihood = count_distribution.log_prob(tf.sparse.to_dense(counts))  # 'counts' datos reales
    count_log_likelihood = tf.reduce_sum(count_log_likelihood, axis=[1, 2])

    # Adjust for the fact that we're only using a minibatch.
    count_log_likelihood = count_log_likelihood * (num_documents / batch_size)

    #### 3. Compute entropy (de distribuciones de reparametrización que optimizan su loc y scale)
    document_entropy = -tf.reduce_sum(document_distribution.log_prob(document_samples), axis=[1, 2])
    objective_topic_entropy = -tf.reduce_sum(objective_topic_distribution.log_prob(objective_topic_samples),
                                             axis=[1, 2])
    ideological_topic_entropy = -tf.reduce_sum(ideological_topic_distribution.log_prob(ideological_topic_samples),
                                               axis=[1, 2])
    ideal_point_entropy = -tf.reduce_sum(ideal_point_distribution.log_prob(ideal_point_samples), axis=1)

    entropy = (document_entropy +
               objective_topic_entropy +
               ideological_topic_entropy +
               ideal_point_entropy)

    # Compute ELBO
    '''
    log_prior: 
    count_log_likelihood: log_likelihood de que counts [batch, words] venga de la distribución Poisson generada por la fórmula que genera rate
    entropy: 
    '''
    elbo = log_prior + count_log_likelihood + entropy
    elbo = tf.reduce_mean(elbo)

    return elbo


def get_log_prior(samples, prior):
    """Return log prior of sampled Gaussians.

    Args:
      samples: A `Tensor` with shape `[num_samples, :, :]`.
      prior: String representing prior distribution.

    Returns:
      log_prior: A `Tensor` with shape `[num_samples]`, with the log priors
        summed across latent dimensions.
    """
    if prior == 'normal':
        prior_distribution = tfp.distributions.Normal(loc=0., scale=1.)
    elif prior == 'gamma':
        prior_distribution = tfp.distributions.Gamma(concentration=0.3, rate=0.3)

    log_prior = tf.reduce_sum(prior_distribution.log_prob(samples), axis=[1, 2])
    return log_prior


def train_model(project_dir, num_topics=15, batch_size=128, max_steps = 100000, print_steps = 100):

    # Seeds
    tf.set_random_seed(42)
    random_state = np.random.RandomState(42)

    # Directorios
    data_dir = os.path.join(project_dir, 'clean')
    save_dir = os.path.join(project_dir, f'fits_{num_topics}') # directorio donde se guardarán los outputs del modelo

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pre_initialize_parameters = True

    # Valores iniciales para document_distribution
    counts = sparse.load_npz(os.path.join(data_dir, 'counts.npz'))
    num_documents, num_words = counts.shape
    if pre_initialize_parameters:
        nmf_model = NMF(n_components=num_topics,
                      init='random',
                      random_state=0,
                      max_iter=500)
        # Add offset to make sure none are zero.
        initial_document_loc = np.float32(nmf_model.fit_transform(counts) + 1e-3)
        initial_objective_topic_loc = np.float32(nmf_model.components_ + 1e-3)

    else:
        initial_document_loc = np.float32(
            np.exp(random_state.randn(num_documents, num_topics)))
        initial_objective_topic_loc = np.float32(
            np.exp(random_state.randn(num_topics, num_words)))

    # Definir batch_size
    # El batch_size indica cuántos documentos procesará el modelo en cada iteración
    # Le puse 128 porque antes estaba trabajando con un dataframe de 600
    # intervenciones parlamentarias (pocas) pero generalmente trabajo con un batch_size
    # de 1024
    #batch_size = 1024 # Prueben que sucede si se aumenta el batch_size.

    # Crear iterador para entrenamiento
    (iterator, author_weights, vocabulary, author_map,
     num_documents, num_words, num_authors) = tbip.build_input_pipeline(
          data_dir,
          batch_size,
          random_state,
          counts_transformation='nothing')
    document_indices, counts, author_indices = iterator.get_next()

    # Crear distribuciones a optimizar con Tensorflow
    # Create Lognormal variational family for document intensities (theta).
    document_loc = tf.get_variable(
        "document_loc",
        initializer=tf.constant(np.log(initial_document_loc)))
    document_scale_logit = tf.get_variable(
        "document_scale_logit",
        shape=[num_documents, num_topics],
        initializer=tf.initializers.random_normal(mean=-2, stddev=1.),
        dtype=tf.float32)
    document_scale = tf.nn.softplus(document_scale_logit)
    document_distribution = tfp.distributions.LogNormal(
        loc=document_loc,
        scale=document_scale)

    # Create Lognormal variational family for objective topics (beta).
    objective_topic_loc = tf.get_variable(
        "objective_topic_loc",
        initializer=tf.constant(np.log(initial_objective_topic_loc)))
    objective_topic_scale_logit = tf.get_variable(
        "objective_topic_scale_logit",
        shape=[num_topics, num_words],
        initializer=tf.initializers.random_normal(mean=-2, stddev=1.),
        dtype=tf.float32)
    objective_topic_scale = tf.nn.softplus(objective_topic_scale_logit)
    objective_topic_distribution = tfp.distributions.LogNormal(
        loc=objective_topic_loc,
        scale=objective_topic_scale)

    # Create Gaussian variational family for ideological topics (eta).
    ideological_topic_loc = tf.get_variable(
        "ideological_topic_loc",
        shape=[num_topics, num_words],
        dtype=tf.float32)
    ideological_topic_scale_logit = tf.get_variable(
        "ideological_topic_scale_logit",
        shape=[num_topics, num_words],
        dtype=tf.float32)
    ideological_topic_scale = tf.nn.softplus(ideological_topic_scale_logit)
    ideological_topic_distribution = tfp.distributions.Normal(
        loc=ideological_topic_loc,
        scale=ideological_topic_scale)

    # Create Gaussian variational family for ideal points (x).
    ideal_point_loc = tf.get_variable(
        "ideal_point_loc",
        shape=[num_authors],
        dtype=tf.float32)
    ideal_point_scale_logit = tf.get_variable(
        "ideal_point_scale_logit",
        initializer=tf.initializers.random_normal(mean=0, stddev=1.),
        shape=[num_authors],
        dtype=tf.float32)
    ideal_point_scale = tf.nn.softplus(ideal_point_scale_logit)
    ideal_point_distribution = tfp.distributions.Normal(
        loc=ideal_point_loc,
        scale=ideal_point_scale)


    # Approximate ELBO.
    elbo = get_elbo(counts,
                    document_indices,
                    author_indices,
                    author_weights,
                    objective_topic_distribution,
                    document_distribution,
                    ideological_topic_distribution,
                    ideal_point_distribution,
                    num_documents,
                    batch_size)

    # Define loss as -elbo
    loss = -elbo


    # Configurar optimización
    learning_rate = 0.001
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optim.minimize(loss)

    neutral_mean = (objective_topic_loc +
                    (objective_topic_scale ** 2 / 2))
    positive_mean = (objective_topic_loc +
                     ideological_topic_loc +
                     (objective_topic_scale ** 2 +
                      ideological_topic_scale ** 2) / 2)
    negative_mean = (objective_topic_loc -
                     ideological_topic_loc +
                     (objective_topic_scale ** 2 +
                      ideological_topic_scale ** 2) / 2)


    # Entrenar el modelo
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    start_time = time.time()
    elbo_history = []

    for step in range(max_steps):
      (_, elbo_val) = sess.run([train_op, elbo])
      elbo_history.append(elbo_val)

      if step % print_steps == 0 or step == max_steps - 1:
        duration = (time.time() - start_time) / (step + 1)
        print("Step: {:>3d} ELBO: {:.3f} ({:.3f} sec/step)".format(
            step, elbo_val, duration))

      if (step + 1) % 2500 == 0 or step == max_steps - 1:
        (document_loc_val,
         document_scale_val,
         objective_topic_loc_val, objective_topic_scale_val,
         ideological_topic_loc_val, ideological_topic_scale_val,
         ideal_point_loc_val, ideal_point_scale_val,
         positive_mean_val, negative_mean_val,
         neutral_mean_val) = sess.run([document_loc, document_scale,
                                       objective_topic_loc, objective_topic_scale,
                                                           ideological_topic_loc,
                                                           ideological_topic_scale,
                                                           ideal_point_loc,
                                                           ideal_point_scale,
                                                           positive_mean,
                                                           negative_mean,
                                                           neutral_mean])

        np.save(os.path.join(save_dir, 'document_loc.npy'), document_loc_val)
        np.save(os.path.join(save_dir, 'document_scale.npy'), document_scale_val)
        np.save(os.path.join(save_dir, 'objective_topic_loc.npy'), objective_topic_loc_val)
        np.save(os.path.join(save_dir, 'objective_topic_scale.npy'), objective_topic_scale_val)
        np.save(os.path.join(save_dir, 'ideological_topic_loc.npy'), ideological_topic_loc_val)
        np.save(os.path.join(save_dir, 'ideological_topic_scale.npy'), ideological_topic_scale_val)
        np.save(os.path.join(save_dir, 'ideal_point_loc.npy'), ideal_point_loc_val)
        np.save(os.path.join(save_dir, 'ideal_point_scale.npy'), ideal_point_scale_val)
        np.save(os.path.join(save_dir, 'elbo_history.npy'), elbo_history)

        print(get_topics(neutral_mean_val,
                        negative_mean_val,
                        positive_mean_val,
                        vocabulary))