from pyro.util import ignore_jit_warnings
from .messenger import Messenger


class BroadcastMessenger(Messenger):
    """
    `BroadcastMessenger` automatically broadcasts the batch shape of
    the stochastic function at a sample site when inside a single
    or nested plate context. The existing `batch_shape` must be
    broadcastable with the size of the :class:`~pyro.plate`
    contexts installed in the `cond_indep_stack`.
    """
    @staticmethod
    @ignore_jit_warnings(["Converting a tensor to a Python boolean"])
    def _pyro_sample(msg):
        """
        :param msg: current message at a trace site.
        """
        if msg["done"] or msg["type"] != "sample":
            return

        dist = msg["fn"]
        actual_batch_shape = getattr(dist, "batch_shape", None)
        #print("actual_batch_shape", actual_batch_shape)
        if actual_batch_shape is not None:
            target_batch_shape = [None if size == 1 else size
                                  for size in actual_batch_shape]
            #print("target_batch_shape_1", target_batch_shape)
            for f in msg["cond_indep_stack"]:
                if f.dim is None or f.size == -1:
                    continue
                assert f.dim < 0
                target_batch_shape = [None] * (-f.dim - len(target_batch_shape)) + target_batch_shape
                #print("target_batch_shape_2", target_batch_shape)
                if target_batch_shape[f.dim] is not None and target_batch_shape[f.dim] != f.size:
                    raise ValueError("Shape mismatch inside plate('{}') at site {} dim {}, {} vs {}".format(
                        f.name, msg['name'], f.dim, f.size, target_batch_shape[f.dim]))
                #print("f.dim", f.dim)
                #print("f.size", f.size)
                target_batch_shape[f.dim] = f.size
                #print("target_batch_shape_3", target_batch_shape)
            # Starting from the right, if expected size is None at an index,
            # set it to the actual size if it exists, else 1.
            for i in range(-len(target_batch_shape) + 1, 1):
                if target_batch_shape[i] is None:
                    #print("yes")
                    target_batch_shape[i] = actual_batch_shape[i] if len(actual_batch_shape) >= -i else 1
            #print("target_batch_shape_4", target_batch_shape)
            msg["fn"] = msg["fn"].expand(target_batch_shape)
