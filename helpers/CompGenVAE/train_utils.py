import torch
import numpy as np
import tqdm


from logging import getLogger
logger = getLogger("CompGenVAE_LOSS_CALCULATOR")
logger.setLevel("DEBUG")




class FocalLoss(torch.nn.Module):
    """
    Focal Loss for binary classification problems to address class imbalance.

    Focal Loss=−α(1−pt)^γ * log(pt)

    Attributes:
    -----------
    alpha : float
        A weighting factor for balancing the importance of positive/negative classes.
        Typically in the range [0, 1]. Higher values of alpha give more weight to the positive class.

    gamma : float
        The focusing parameter. Gamma >= 0. Reduces the relative loss for well-classified examples,
        putting more focus on hard, misclassified examples. Higher values of gamma make the model focus more
        on hard examples.

    Methods:
    --------
    forward(inputs, targets):
        Compute the focal loss for given inputs and targets.

    Parameters:
    -----------
    inputs : torch.Tensor
        The logits (raw model outputs) of shape (N, *) where * means any number of additional dimensions.
        For binary classification, this is typically of shape (N, 1).

    targets : torch.Tensor
        The ground truth values (labels) with the same shape as inputs.

    Returns:
    --------
    torch.Tensor
        The computed focal loss.

    Example:
    --------
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    loss = criterion(logits, targets)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for computing the focal loss.
        """
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Computes the probability of the correct class (Prevents nans when probability 0)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss

        if self.reduction.lower() == 'mean':
            return torch.mean(F_loss)
        elif self.reduction.lower() == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def generate_beta_curve(n_epochs, period_epochs, rise_ratio, start_first_rise_at_epoch=0):
    """
    Generate a beta curve for the given parameters

    Args:
        n_epochs:            The number of epochs to generate the curve for
        period_epochs:       The period of the curve in epochs (for multiple cycles)
        rise_ratio:         The ratio of the period to be used for the rising part of the curve
        start_first_rise_at_epoch:  The epoch to start the first rise at (useful for warmup)

    Returns:

    """
    def f(x, K):
        if x == 0:
            return 0
        elif x == K:
            return 1
        else:
            return 1 / (1 + np.exp(-10 * (x - K / 2) / K))

    def generate_rising_curve(K):
        curve = []
        for i in range(K):
            curve.append(f(i, K - 1))
        return np.array(curve)

    def generate_single_beta_cycle(period, rise_ratio):
        cycle = np.ones(period)

        curve_steps_in_epochs = int(period * rise_ratio)

        rising_curve = generate_rising_curve(curve_steps_in_epochs)

        cycle[:rising_curve.shape[0]] = rising_curve[:cycle.shape[0]]

        return cycle

    beta_curve = np.zeros((start_first_rise_at_epoch))
    effective_epochs = n_epochs - start_first_rise_at_epoch
    n_cycles = np.ceil(effective_epochs / period_epochs)

    single_cycle = generate_single_beta_cycle(period_epochs, rise_ratio)

    for c in np.arange(n_cycles):
        beta_curve = np.append(beta_curve, single_cycle)

    return beta_curve[:n_epochs]


def calculate_hit_loss(hit_logits, hit_targets, hit_loss_function):
    assert isinstance(hit_loss_function, torch.nn.BCEWithLogitsLoss) or isinstance(hit_loss_function, FocalLoss)
    loss_h = hit_loss_function(hit_logits, hit_targets)           # batch, time steps, voices
    return loss_h       # batch_size,  time_steps, n_voices


def calculate_velocity_loss(vel_logits, vel_targets, vel_loss_function):
    if isinstance(vel_loss_function, torch.nn.BCEWithLogitsLoss):
        loss_v = vel_loss_function(vel_logits, vel_targets)
    else:
        raise NotImplementedError(f"the vel_loss_function {vel_loss_function} is not implemented")

    return loss_v       # batch_size,  time_steps, n_voices


def calculate_offset_loss(offset_logits, offset_targets, offset_loss_function):
    if isinstance(offset_loss_function, torch.nn.BCEWithLogitsLoss):
        # here the offsets MUST be in the range of [0, 1]. Our existing targets are from [-0.5, 0.5].
        # So we need to shift them to [0, 1] range by adding 0.5
        loss_o = offset_loss_function(offset_logits, offset_targets+0.5)
    else:
        raise NotImplementedError(f"the offset_loss_function {offset_loss_function} is not implemented")

    return loss_o           # batch_size,  time_steps, n_voices


def calculate_complexity_loss(complexity_logits, complexity_targets, complexity_loss_function):
    if len(complexity_targets.shape) == 1:
        complexity_targets = complexity_targets.unsqueeze(1)

    if isinstance(complexity_loss_function, torch.nn.BCEWithLogitsLoss):
        loss_c = complexity_loss_function(complexity_logits, complexity_targets)
    else:
        raise NotImplementedError(f"the complexity_loss_function {complexity_loss_function} is not implemented")

    return loss_c


def calculate_genre_loss(genre_logits, genre_targets, genre_loss_function):

    if isinstance(genre_loss_function, torch.nn.CrossEntropyLoss):
        loss_g = genre_loss_function(genre_logits, genre_targets)
    else:
        raise NotImplementedError(f"the genre_loss_function {genre_loss_function} is not implemented")
    return loss_g


def calculate_kld_loss(mu, log_var):
    """ calculate the KLD loss for the given mu and log_var values against a standard normal distribution
    :param mu:  (torch.Tensor)  the mean values of the latent space
    :param log_var: (torch.Tensor)  the log variance values of the latent space
    :return:    kld_loss (torch.Tensor)  the KLD loss value (unreduced) shape: (batch_size,  time_steps, n_voices)

    """
    mu = mu.view(mu.shape[0], -1)
    log_var = log_var.view(log_var.shape[0], -1)
    kld_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

    return kld_loss.mean()     # batch_size,  time_steps, n_voices


def batch_loop(dataloader_, model, hit_loss_fn, velocity_loss_fn,  offset_loss_fn,
               complexity_loss_fn, genre_loss_fn,
               device, optimizer=None, starting_step=None, kl_beta=1.0,
               teacher_forcing_ratio=0.5):

    """
    This function iteratively loops over the given dataloader and calculates the loss for each batch. If an optimizer is
    provided, it will also perform the backward pass and update the model parameters. The loss values are accumulated
    and returned at the end of the loop.

    **Can be used for both training and testing. In testing however, backpropagation will not be performed**


    :param dataloader_:     (torch.utils.data.DataLoader)  dataloader for the dataset
    :param model:  (ComplexityGenreVAE)  the model
    :param hit_loss_fn:     (str)  "bce"
    :param velocity_loss_fn:    (str)  "bce"
    :param offset_loss_fn:  (str)  "bce"
    :param complexity_loss_fn:  (str)  "bce"
    :param genre_loss_fn:  (str)  "cross_entropy"
    :param device:  (torch.device)  the device to use for the model
    :param optimizer:   (torch.optim.Optimizer)  the optimizer to use for the model
    :param starting_step:   (int)  the starting step for the optimizer
    :param kl_beta: (float)  the beta value for the KLD loss
    :param teacher_forcing_ratio: (float)  the ratio of teacher forcing to use
    :return:    (dict)  a dictionary containing the loss values for the current batch

                metrics = {
                    "loss_total": np.mean(loss_total),
                    "loss_h": np.mean(loss_h),
                    "loss_v": np.mean(loss_v),
                    "loss_o": np.mean(loss_o),
                    "loss_KL": np.mean(loss_KL)}

                (int)  the current step of the optimizer (if provided)
    """
    # Prepare the metric trackers for the new epoch
    # ------------------------------------------------------------------------------------------
    loss_total, loss_recon, loss_h, loss_v, loss_o, loss_KL, loss_KL_beta_scaled = [], [], [], [], [], [], []
    loss_complexity, loss_genre = [], []

    # Iterate over batches
    # ------------------------------------------------------------------------------------------
    total_batches = len(dataloader_)
    for batch_count, (data_) in (pbar := tqdm.tqdm(enumerate(dataloader_), total=total_batches)):

        inputs_ = data_[0]
        complexities_ = data_[1]
        genres_one_hot_ = data_[3]

        genre_true_logits = genres_one_hot_.to(device) if genres_one_hot_.device.type != device else genres_one_hot_
        genre_true_logits = torch.log(genre_true_logits + 1e-4)  # convert to logits
        genre_targets = data_[4].type(torch.LongTensor)
        # convert targets to int

        outputs_ = data_[5]
        # index = data_[6]

        # Move data to GPU if available
        # ---------------------------------------------------------------------------------------
        inputs = inputs_.to(device) if inputs_.device.type!= device else inputs_
        complexities = complexities_.to(device) if complexities_.device.type!= device else complexities_
        genres_targets = genre_targets.to(device) if genre_targets.device.type!= device else genre_targets
        outputs = outputs_.to(device) if outputs_.device.type!= device else outputs_

        # Forward pass
        # ---------------------------------------------------------------------------------------
        # print(f"inputs: {inputs.shape}")
        # print(f"densities: {densities.shape}")
        mu, log_var, latent_z, memory = model.encodeLatent(flat_hvo_groove=inputs)
        genre_logits = model.encodeGenre(memory)
        complexity_logits = model.encodeComplexity(memory)

        batch_loss_c = calculate_complexity_loss(
            complexity_logits=complexity_logits, complexity_targets=complexities,
            complexity_loss_function=complexity_loss_fn)

        batch_loss_g = calculate_genre_loss(
            genre_logits=genre_logits, genre_targets=genres_targets, genre_loss_function=genre_loss_fn)

        # if teacher_forcing_ratio > 0, replace genre_logits with genre_true_logits and complexity_logits with complexities
        if np.random.rand() <= (teacher_forcing_ratio + 1e-4):
            genre_logits = genre_true_logits
            complexity_logits = complexities[:, None]
            complexity_logits = torch.log(complexity_logits / (1 - complexity_logits) + 1e-4)  # convert to logits

        # Decode
        h_logits, v_logits, o_logits, hvo_logits = model.decode_with_logits(
            latent_z=latent_z,
            genre_logits=genre_logits,
            complexity_logits=complexity_logits
        )

        # Prepare targets for loss calculation
        h_targets, v_targets, o_targets = torch.split(outputs, int(outputs.shape[2] / 3), 2)

        # Compute losses
        # ---------------------------------------------------------------------------------------
        batch_loss_h = calculate_hit_loss(
            hit_logits=h_logits, hit_targets=h_targets, hit_loss_function=hit_loss_fn)

        batch_loss_v = calculate_velocity_loss(
            vel_logits=v_logits, vel_targets=v_targets, vel_loss_function=velocity_loss_fn)

        batch_loss_o = calculate_offset_loss(
            offset_logits=o_logits, offset_targets=o_targets, offset_loss_function=offset_loss_fn)

        batch_loss_KL = calculate_kld_loss(mu, log_var)

        batch_loss_KL_Beta_Scaled = batch_loss_KL * kl_beta

        batch_loss_recon = (batch_loss_h + batch_loss_v + batch_loss_o)
        batch_loss_total = (batch_loss_recon + batch_loss_KL_Beta_Scaled + batch_loss_c)

        # Backpropagation and optimization step (if training)
        # ---------------------------------------------------------------------------------------
        if optimizer is not None:
            optimizer.zero_grad()
            batch_loss_total.backward(retain_graph=True)
            batch_loss_g.backward(retain_graph=True)
            batch_loss_c.backward(retain_graph=True)
            optimizer.step()

        # Update the per batch loss trackers
        # -----------------------------------------------------------------
        loss_h.append(batch_loss_h.item())
        loss_v.append(batch_loss_v.item())
        loss_o.append(batch_loss_o.item())
        loss_complexity.append(batch_loss_c.item())
        loss_genre.append(batch_loss_g.item())
        loss_total.append(batch_loss_total.item())
        loss_recon.append(batch_loss_recon.item())
        loss_KL.append(batch_loss_KL.item())
        loss_KL_beta_scaled.append(batch_loss_KL_Beta_Scaled.item())

        # Update the progress bar
        # ---------------------------------------------------------------------------------------
        pbar.set_postfix(
            {"l_total_recon_kl": f"{np.mean(loss_total):.4f}",
                "l_c": f"{np.mean(loss_complexity):.4f}",
                "l_g": f"{np.mean(loss_genre):.4f}",
                "l_recon": f"{np.mean(loss_recon):.4f}"
                })

        # Increment the step counter
        # ---------------------------------------------------------------------------------------
        if starting_step is not None:
            starting_step += 1

    # empty gpu cache if cuda
    if device != 'cpu':
        torch.cuda.empty_cache()

    metrics = {
        "loss_total_rec_w_kl": np.mean(loss_total),
        "loss_h": np.mean(loss_h),
        "loss_v": np.mean(loss_v),
        "loss_o": np.mean(loss_o),
        "loss_complexity": np.mean(loss_complexity),
        "loss_genre": np.mean(loss_genre),
        "loss_KL": np.mean(loss_KL),
        "loss_KL_beta_scaled": np.mean(loss_KL_beta_scaled),
        "loss_recon": np.mean(loss_recon)
    }

    if starting_step is not None:
        return metrics, starting_step
    else:
        return metrics


def train_loop(train_dataloader, model, optimizer, hit_loss_fn, velocity_loss_fn, offset_loss_fn, 
               complexity_loss_fn, genre_loss_fn,
               device, starting_step, kl_beta=1,
               teacher_forcing_ratio=0.5):
    """
    This function performs the training loop for the given model and dataloader. It will iterate over the dataloader
    and perform the forward and backward pass for each batch. The loss values are accumulated and the average is
    returned at the end of the loop.

    :param train_dataloader:    (torch.utils.data.DataLoader)  dataloader for the training dataset
    :param model:  (GrooveTransformerVAE)  the model
    :param optimizer:  (torch.optim.Optimizer)  the optimizer to use for the model (sgd or adam)
    :param hit_loss_fn:     ("dice" or torch.nn.BCEWithLogitsLoss)
    :param velocity_loss_fn:  (torch.nn.BCEWithLogitsLoss)
    :param offset_loss_fn:      (torch.nn.BCEWithLogitsLoss)
    :param complexity_loss_fn:  (torch.nn.BCEWithLogitsLoss)
    :param genre_loss_fn:  (torch.nn.CrossEntropyLoss)
    :param device:  (str)  the device to use for the model
    :param starting_step:   (int)  the starting step for the optimizer
    :param kl_beta: (float)  the beta value for the KL loss (maybe flat or annealed)
    :param teacher_forcing_ratio: (float)  the ratio of teacher forcing to use

    :return:    (dict)  a dictionary containing the loss values for the current batch

            metrics = {
                    "train/loss_total": np.mean(loss_total),
                    "train/loss_h": np.mean(loss_h),
                    "train/loss_v": np.mean(loss_v),
                    "train/loss_o": np.mean(loss_o),
                    "train/loss_KL": np.mean(loss_KL)}
    """
    # ensure model is in training mode
    if model.training is False:
        logger.warning("Model is not in training mode. Setting to training mode.")
        model.train()

    # Run the batch loop
    metrics, starting_step = batch_loop(
        dataloader_=train_dataloader,
        model=model,
        hit_loss_fn=hit_loss_fn,
        velocity_loss_fn=velocity_loss_fn,
        offset_loss_fn=offset_loss_fn,
        complexity_loss_fn=complexity_loss_fn,
        genre_loss_fn=genre_loss_fn,
        device=device,
        optimizer=optimizer,
        starting_step=starting_step,
        kl_beta=kl_beta,
        teacher_forcing_ratio=teacher_forcing_ratio)

    metrics = {f"Loss_Criteria/{key}_train": value for key, value in sorted(metrics.items())}
    return metrics, starting_step


def test_loop(test_dataloader, model, hit_loss_fn, velocity_loss_fn, offset_loss_fn, 
              complexity_loss_fn, genre_loss_fn,
              device, kl_beta=1,
              teacher_forcing_ratio=0.5):
    """
    This function performs the test loop for the given model and dataloader. It will iterate over the dataloader
    and perform the forward pass for each batch. The loss values are accumulated and the average is returned at the end
    of the loop.

    :param test_dataloader:   (torch.utils.data.DataLoader)  dataloader for the test dataset
    :param model:  (GrooveTransformerVAE)  the model
    :param hit_loss_fn:     ("dice" or torch.nn.BCEWithLogitsLoss)
    :param velocity_loss_fn:    (torch.nn.BCEWithLogitsLoss)
    :param offset_loss_fn:    (torch.nn.BCEWithLogitsLoss)
    :param complexity_loss_fn:  (torch.nn.BCEWithLogitsLoss)
    :param genre_loss_fn:  (torch.nn.CrossEntropyLoss)
    :param device:  (str)  the device to use for the model
    :param kl_beta: (float)  the beta value for the KL loss
    :param teacher_forcing_ratio: (float)  the ratio of teacher forcing to use

    :return:   (dict)  a dictionary containing the loss values for the current batch

            metrics = {
                    "test/loss_total": np.mean(loss_total),
                    "test/loss_h": np.mean(loss_h),
                    "test/loss_v": np.mean(loss_v),
                    "test/loss_o": np.mean(loss_o),
                    "test/loss_KL": np.mean(loss_KL)}
    """
    # ensure model is in eval mode
    if model.training is True:
        logger.warning("Model is not in eval mode. Setting to eval mode.")
        model.eval()

    with torch.no_grad():
        # Run the batch loop
        metrics = batch_loop(
            dataloader_=test_dataloader,
            model=model,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            complexity_loss_fn=complexity_loss_fn,
            genre_loss_fn=genre_loss_fn,
            device=device,
            optimizer=None,
            kl_beta=kl_beta,
            teacher_forcing_ratio=teacher_forcing_ratio)

    metrics = {f"Loss_Criteria/{key}_test": value for key, value in sorted(metrics.items())}

    return metrics


