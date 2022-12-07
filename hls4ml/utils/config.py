from __future__ import print_function
import json

import hls4ml

def create_config(output_dir='my-hls-test', project_name='myproject',
    backend='Vivado', **kwargs):

    backend_list = hls4ml.backends.get_available_backends()
    if backend.lower() not in backend_list:
        raise Exception('Unknown backend: {}'.format(backend))

    backend = hls4ml.backends.get_backend(backend)

    backend_config = backend.create_initial_config(**kwargs)

    config = {}
    config['OutputDir'] = output_dir
    config['ProjectName'] = project_name
    config['Backend'] = backend.name
    config.update(backend_config)

    return config

def _get_precision_from_quantizer(quantizer):
    import qkeras
    if isinstance(quantizer, str):
        quantizer_obj = qkeras.get_quantizer(quantizer)
        quantizer = {}
        # Some activations are classes with get_config method
        if hasattr(quantizer_obj, 'get_config'):
            quantizer['class_name'] = quantizer_obj.__class__.__name__
            quantizer['config'] = quantizer_obj.get_config()
        # Some activations are just functions
        else: 
            quantizer['class_name'] = quantizer_obj.__name__

    supported_quantizers = ['quantized_bits', 'quantized_relu', 'quantized_tanh', 'quantized_po2', 'quantized_relu_po2']
    signed = True
    if quantizer['class_name'] in supported_quantizers:
        bits = int(quantizer['config']['bits'])
        # if integer isn't specified, it should be the same as bits
        integer = int(quantizer['config'].get('integer', bits-1)) + 1
        if quantizer['class_name'] == 'quantized_relu':
            signed = False
            integer -= 1
    elif quantizer['class_name'] in ['binary', 'stochastic_binary', 'binary_tanh']:
        bits = 2
        integer = 2
    
    elif quantizer['class_name'] in ['ternary', 'stochastic_ternary', 'ternary_tanh']:
        bits = 2
        integer = 2
    else:
        raise Exception('ERROR: Unsupported quantizer: {}'.format(quantizer['class_name']))

    decimal = bits - integer

    if decimal > 0:
        return hls4ml.model.types.FixedPrecisionType(width=bits, integer=integer, signed=signed)
    else:
        return hls4ml.model.types.IntegerPrecisionType(width=integer, signed=signed)

def config_from_keras_model(model, granularity='model', backend=None, default_precision='fixed<16,6>', default_reuse_factor=1):
    """Create an HLS conversion config given the Keras model.

    This function serves as the initial step in creating the custom conversion configuration.
    Users are advised to inspect the returned object to tweak the conversion configuration.
    The return object can be passed as `hls_config` parameter to `convert_from_keras_model`.

    Args:
        model: Keras model
        granularity (str, optional): Granularity of the created config. Defaults to 'model'.
            Can be set to 'model', 'type' and 'layer'.

            Granularity can be used to generate a more verbose config that can be fine-tuned.
            The default granulrity ('model') will generate config keys that apply to the whole
            model, so changes to the keys will affect the entire model. 'type' granularity will
            generate config keys that affect all layers of a given type, while the 'name' granularity
            will generate config keys for every layer separately, allowing for highly specific
            configuration tweaks.
        backend(str, optional): Name of the backend to use
        default_precision (str, optional): Default precision to use. Defaults to 'fixed<16,6>'.
        default_reuse_factor (int, optional): Default reuse factor. Defaults to 1.

    Raises:
        Exception: If Keras model has layers not supported by hls4ml.

    Returns:
        [dict]: The created config.
    """
    if granularity.lower() not in ['model', 'type', 'name']:
        raise Exception('Invalid configuration granularity specified, expected "model", "type" or "name" got "{}"'.format(granularity))

    if backend is not None:
        backend = hls4ml.backends.get_backend(backend)

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    if isinstance(model, dict):
        model_arch = model
    else:
        model_arch = json.loads(model.to_json())

    reader = hls4ml.converters.KerasModelReader(model)

    layer_list, _, _ = hls4ml.converters.parse_keras_model(model_arch, reader)

    def make_layer_config(layer):
        cls_name = layer['class_name']
        if 'config' in layer.keys():
            if 'activation' in layer['config'].keys():
                if layer['config']['activation'] == 'softmax':
                    cls_name = 'Softmax'
        
        layer_cls = hls4ml.model.layers.layer_map[cls_name]
        if backend is not None:
            layer_cls = backend.create_layer_class(layer_cls)
        
        layer_config = {}

        config_attrs = [a for a in layer_cls.expected_attributes if a.configurable]
        for attr in config_attrs:
            if isinstance(attr, hls4ml.model.attributes.TypeAttribute):
                precision_cfg = layer_config.setdefault('Precision', {})
                name = attr.name
                if name.endswith('_t'):
                    name = name[:-2]
                if attr.default is None:
                    precision_cfg[name] = default_precision
                else:
                    precision_cfg[name] = str(attr.default)
            else:
                if attr.default is not None:
                    layer_config[attr.config_name] = attr.default
            

        quantizers = { qname: qclass for qname, qclass in layer.items() if 'quantizer' in qname}
        for qname, qclass in quantizers.items():
            pname = qname.lower().split('_quantizer')[0]
            if pname == 'activation': pname = 'result'
            if isinstance(qclass, dict):
                precision = _get_precision_from_quantizer(qclass)
            else:
                precision = qclass.hls_type
            #TODO In the next version of this function, these should not be exposed to user to tweak
            layer_config['Precision'][pname] = str(precision)

        if layer['class_name'] in ['GarNet', 'GarNetStack']:
            ## Following code copy-pasted from hls4ml.model.hls_layers - can we factor out commonalities between the two modules?

            ## Define default precisions for various internal arrays (can be overridden from the config file)
            import math
            log2_reuse = int(math.log(default_reuse_factor, 2.))
            n_vertices_width = int(math.log(layer['n_vertices'], 2.))

            # We always give 10 digits for the subintegral part
            fwidth = 10
            # Integral precision for aggr_t depends on how large the temporary sum for weighed feature mean will be
            aggr_intw = max(log2_reuse, n_vertices_width - log2_reuse) + 3 # safety factor 2**3
            aggr_w = aggr_intw + fwidth
            # edge_weight_aggr_t does not need the safety factor
            ew_aggr_intw = aggr_intw - 3
            ew_aggr_w = ew_aggr_intw + fwidth

            layer_config['Precision'] = {}
            layer_config['Precision']['edge_weight'] = 'ap_ufixed<10,0,AP_TRN,AP_SAT>'
            layer_config['Precision']['edge_weight_aggr'] = 'ap_ufixed<{},{},AP_TRN,AP_SAT>'.format(ew_aggr_w, ew_aggr_intw)
            layer_config['Precision']['aggr'] = 'ap_fixed<{},{},AP_TRN,AP_SAT>'.format(aggr_w, aggr_intw)
            layer_config['Precision']['norm'] = 'ap_ufixed<14,4,AP_TRN,AP_SAT>'

            layer_config['ReuseFactor'] = default_reuse_factor

        elif layer['class_name'] == 'Input':
            dtype = layer['config']['dtype']
            if dtype.startswith('int') or dtype.startswith('uint'):
                typename = dtype[:dtype.index('int') + 3]
                width = int(dtype[dtype.index('int') + 3:])
                layer_config['Precision']['result'] = 'ap_{}<{}>'.format(typename, width)
            # elif bool, q[u]int, ...

        return layer_config

    config = {}

    model_config = {}
    model_config['Precision'] = default_precision
    model_config['ReuseFactor'] = default_reuse_factor
    model_config['Strategy'] = 'Latency'
    model_config['BramFactor'] = 1_000_000_000
    #model_config['Compression'] = False
    model_config['TraceOutput'] = False

    config['Model'] = model_config
    
    if granularity.lower() == 'type':
        type_config = {}
        for layer in layer_list:
            if layer['class_name'] in type_config:
                continue
            layer_config = make_layer_config(layer)
            type_config[layer['class_name']] = layer_config
        
        config['LayerType'] = type_config

    elif granularity.lower() == 'name':
        name_config = {}
        for layer in layer_list:
            layer_config = make_layer_config(layer)
            name_config[layer['name']] = layer_config
        
        config['LayerName'] = name_config

    return config


def config_from_pytorch_model(model, granularity='model', default_precision='ap_fixed<16,6>', default_reuse_factor=1):
    """Generate configuration dictionary from a Pytorch model.
    
    Parameters
    ----------
    model : Pytorch model object.
        Model to be converted to hls model object.
    granularity : string, optional
        How granular you want the configuration to be.
    default_precision : string, optional
        Defines the precsion of your inputs, outputs, weights and biases.
        It is denoted by ap_fixed<X,Y>, where Y is the number of bits representing 
        the signed number above the binary point (i.e. the integer part),
        and X is the total number of bits. Additionally, integers in fixed precision 
        data type (ap_int<N>, where N is a bit-size from 1 to 1024) can also be used. 
    default_reuse_factor : int, optional
        Reuse factor for hls model
        
    Returns
    -------
    config : dict
        configuration dictionary to be used in Pytorch converter.
        
    See Also
    --------
    hls4ml.config_from_keras_model, hls4ml.convert_from_onnx_model
    
    Examples
    --------
    >>> import hls4ml
    >>> config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    >>> hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)

    """
    
    config = {}

    model_config = {}
    model_config['Precision'] = default_precision
    model_config['ReuseFactor'] = default_reuse_factor
    model_config['Strategy'] = 'Latency'

    config['Model'] = model_config
    
    return config


def config_from_onnx_model(model, granularity='model', default_precision='ap_fixed<16,6>', default_reuse_factor=1):
    """Generate configuration dictionary from an ONNX model.
    
    Parameters
    ----------
    model : ONNX model object.
        Model to be converted to hls model object.
    granularity : string, optional
        How granular you want the configuration to be.
    default_precision : string, optional
        Defines the precsion of your inputs, outputs, weights and biases.
        It is denoted by ap_fixed<X,Y>, where Y is the number of bits representing 
        the signed number above the binary point (i.e. the integer part),
        and X is the total number of bits. Additionally, integers in fixed precision 
        data type (ap_int<N>, where N is a bit-size from 1 to 1024) can also be used. 
    default_reuse_factor : int, optional
        Reuse factor for hls model
        
    Returns
    -------
    config : dict
        configuration dictionary to be used in ONNX converter.
        
    See Also
    --------
    hls4ml.config_from_keras_model, hls4ml.convert_from_pytorch_model
    
    Examples
    --------
    >>> import hls4ml
    >>> config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    >>> hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)

    """
    
    config = {}

    model_config = {}
    model_config['Precision'] = default_precision
    model_config['ReuseFactor'] = default_reuse_factor
    model_config['Strategy'] = 'Latency'

    config['Model'] = model_config
    
    return config
