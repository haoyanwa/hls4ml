from hls4ml.model.layers import BatchNormalization, Conv1D, Conv2D, Pooling1D, Pooling2D, SeparableConv1D
from hls4ml.model.optimizer import OptimizerPass


class ChannelsLastConverter(OptimizerPass):
    '''Converts a model from channels_first to channels_last data format by transposing the weights of relevant layers
    and adding a transpose layer for the inputs.'''

    def match(self, node):
        is_channel_first = False
        if node.get_attr('data_format') == "channels_first":
            is_channel_first = True
        return (
            isinstance(node, (Conv1D, Conv2D, BatchNormalization, Pooling1D, Pooling2D, SeparableConv1D))
            and is_channel_first
        )

    def transform(self, model, node):

        # Transpose weight tensor for layers that have one
        if isinstance(node, (Conv1D, Conv2D, SeparableConv1D)):
            weights_channels_last = node.get_weights('weight').data.transpose()
            node.get_weights('weight').data = weights_channels_last

        # Adjust output shape
        outshape = node.get_output_variable().shape

        if isinstance(node, (Conv1D, Pooling1D, SeparableConv1D)):
            shape = [outshape[1], outshape[0]]
            dims = [f'N_OUTPUTS_{node.get_attr("index")}', f'N_FILT_{node.get_attr("index")}']
            perm = [1, 0]
            perm_out = [1, 0]
        elif isinstance(node, (Conv2D, Pooling2D)):
            shape = [outshape[1], outshape[2], outshape[0]]
            dims = [
                f'OUT_HEIGHT_{node.get_attr("index")}',
                f'OUT_WIDTH_{node.get_attr("index")}',
                f'N_FILT_{node.get_attr("index")}',
            ]
            perm = [1, 2, 0]
            perm_out = [2, 0, 1]
        node.add_output_variable(shape, dims)
        node.set_attr('data_format', 'channels_last')

        # Add transpose for input layer
        input = node.get_input_node().name
        attributes = {'perm': perm}

        transpose_node = model.make_node('Transpose', f'transpose_input_for_{node.get_attr("name")}', attributes, [input])
        transpose_node.set_attr('name', f'transpose_input_for_{node.get_attr("name")}')
        model.insert_node(transpose_node)

        # Add transpose for output
        input = node.name
        attributes = {'perm': perm_out}

        transpose_output_node = model.make_node(
            'Transpose', f'transpose_output_for_{node.get_attr("name")}', attributes, [input]
        )
        transpose_output_node.set_attr('name', f'transpose_output_for_{node.get_attr("name")}')
        model.insert_node(transpose_output_node)

        return True


class ChannelsLastConversionCleaner(OptimizerPass):
    '''Detects and removes unnecessary transpose layers introduced in the conversion to channels_last.'''

    def match(self, node):
        if 'transpose_input_for' in node.name and 'transpose_output_for' in node.get_input_node().name:
            return True

    def transform(self, model, node):

        input_node = node.get_input_node()
        model.remove_node(input_node)
        model.remove_node(node)

        return True