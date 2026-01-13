# Dense + Activation fusion pass for the AIE backend.

from hls4ml.model.optimizer import OptimizerPass
from ..ir import get_backend_context, LogicalIR, TraitInstance


class FuseActivationCasts(OptimizerPass):
    """Fuse Dense+Activation pairs (relu or linear) directly in the hls4ml graph."""

    _SUPPORTED = {'relu', 'linear'}

    ## TO DO: FIX  MATCH FOR NEW CASE

    def match(self, node):
        if getattr(node, 'class_name', None) != 'Activation' or len(node.inputs) != 1:
            return False

        act = (node.get_attr('activation', '') or '').lower()
        if act not in self._SUPPORTED:
            return False

        prev_node = node.get_input_node()
        if prev_node is None or getattr(prev_node, 'class_name', None) != 'Dense':
            return False

        return True
    
    def transform(self, model, node):
        ctx = get_backend_context(model)
        graph: LogicalIR = ctx.ir.logical

        for n in graph.nodes:
            if n.name == node.name + '_aie':
                node = n
                break

        input_node = node.inputs[0].producer
        activation = (node.get_attr('activation', '') or '').lower()
        input_node.add_trait(TraitInstance('fused_activation', {'activation': activation}))

        act_quant = node.metadata.setdefault('quant', {})
        input_quant = input_node.metadata.setdefault('quant', {})
        if not act_quant:
            raise RuntimeError('Quantization information missing. Run quant before invoking downstream passes.')
        input_quant['output_precision'] = act_quant['output_precision']
        
        graph.remove_node(node)

        return True
    
    def _old_match(self, node):
        if getattr(node, 'class_name', None) != 'Activation' or len(node.inputs) != 1:
            return False

        act = (node.get_attr('activation', '') or '').lower()
        if act not in self._SUPPORTED:
            return False

        prev_node = node.get_input_node()
        if prev_node is None or getattr(prev_node, 'class_name', None) != 'Dense':
            return False

        return True

    def _old_transform(self, model, node):
        dense = node.get_input_node()
        activation = (node.get_attr('activation', '') or '').lower()

        in_var = node.get_input_variable()
        out_var = node.get_output_variable()
        in_var.type.precision = out_var.type.precision

        dense.set_attr('aie_fused_activation', activation)
        model.remove_node(node)
        return True
