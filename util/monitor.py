from torch.utils.tensorboard import SummaryWriter

__all__ = ['ProgressMonitor', 'TensorBoardMonitor', 'AverageMeter']


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, fmt='%.6f'):
        self.fmt = fmt
        self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        s = self.fmt % self.avg
        return s


class Monitor:
    """This is an abstract interface for data loggers

    Train monitors log the progress of the training process to some backend.
    This backend can be a file, a web service, or some other means to collect and/or
    display the training
    """

    def __init__(self):
        pass

    def update(self, epoch, step_idx, step_num, prefix, meter_dict):
        raise NotImplementedError


class ProgressMonitor(Monitor):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def update(self, epoch, step_idx, step_num, prefix, meter_dict):
        msg = prefix
        if epoch > -1:
            msg += ' [%d][%5d/%5d]   ' % (epoch, step_idx, int(step_num))
        else:
            msg += ' [%5d/%5d]   ' % (step_idx, int(step_num))
        for k, v in meter_dict.items():
            msg += k + ' '
            if isinstance(v, AverageMeter):
                msg += str(v)
            else:
                msg += '%.6f' % v
            msg += '   '
        self.logger.info(msg)


class TensorBoardMonitor(Monitor):
    def __init__(self, logger, log_dir):
        super().__init__()
        self.writer = SummaryWriter(log_dir / 'tb_runs')
        logger.info('TensorBoard data directory: %s/tb_runs' % log_dir)

    def update(self, epoch, step_idx, step_num, prefix, meter_dict):
        current_step = epoch * step_num + step_idx
        for k, v in meter_dict.items():
            val = v.val if isinstance(v, AverageMeter) else v
            self.writer.add_scalar(prefix + '/' + k, val, current_step)

class GetWeightAndActivation:
    """
    A class used to get weights and activations from specified layers from a Pytorch model.
    """

    def __init__(self, model, layers):
        """
        Args:
            model (nn.Module): the model containing layers to obtain weights and activations from.
            layers (list of strings): a list of layer names to obtain weights and activations from.
                Names are hierarchical, separated by /. For example, If a layer follow a path
                "s1" ---> "pathway0_stem" ---> "conv", the layer path is "s1/pathway0_stem/conv".
        """
        self.model = model
        self.hooks = {}
        self.layers_names = layers
        # eval mode
        self.model.eval()
        self._register_hooks()

    def _get_layer(self, layer_name):
        """
        Return a layer (nn.Module Object) given a hierarchical layer name, separated by /.
        Args:
            layer_name (str): the name of the layer.
        """
        layer_ls = layer_name.split(".")
        prev_module = self.model
        for layer in layer_ls:
            prev_module = prev_module._modules[layer]

        return prev_module

    def _register_single_hook(self, layer_name):
        """
        Register hook to a layer, given layer_name, to obtain activations.
        Args:
            layer_name (str): name of the layer.
        """

        def hook_fn(module, input, output):
            self.hooks[layer_name] = output.clone().detach()

        layer = self._get_layer(layer_name)
        layer.register_forward_hook(hook_fn)

    def _register_hooks(self):
        """
        Register hooks to layers in `self.layers_names`.
        """
        for layer_name in self.layers_names:
            self._register_single_hook(layer_name)

    def get_activations(self, input, bboxes=None):
        """
        Obtain all activations from layers that we register hooks for.
        Args:
            input (tensors, list of tensors): the model input.
            bboxes (Optional): Bouding boxes data that might be required
                by the model.
        Returns:
            activation_dict (Python dictionary): a dictionary of the pair
                {layer_name: list of activations}, where activations are outputs returned
                by the layer.
        """
        if bboxes is not None:
            preds = self.model(input, bboxes)
        else:
            preds = self.model(input)

        activation_dict = {}
        for layer_name, hook in self.hooks.items():
            # list of activations for each instance.
            activation_dict[layer_name] = hook

        return activation_dict, preds

    def get_weights(self):
        """
        Returns weights from registered layers.
        Returns:
            weights (Python dictionary): a dictionary of the pair
            {layer_name: weight}, where weight is the weight tensor.
        """
        weights = {}
        for layer in self.layers_names:
            cur_layer = self._get_layer(layer)
            if hasattr(cur_layer, "weight"):
                weights[layer] = cur_layer.weight.clone().detach()
            # else:
            #     logger.error(
            #         "Layer {} does not have weight attribute.".format(layer)
            #     )
        return weights
