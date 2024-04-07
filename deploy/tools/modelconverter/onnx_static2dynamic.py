import onnx
from onnx.tools import update_model_dims

def changeOnnx2Dynamic(onnx_file, save_path):

    model = onnx.load(onnx_file)
    
    input_shape = [-1, 3, 640, 640]
    input_dims = {}
    for input in model.graph.input:
        input_dims[input.name] = list(input_shape)
    # keep output_dims
    output_dims = {}
    for output in model.graph.output:
        dim = []
        tensor_type = output.type.tensor_type
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    dim.append(d.dim_value)
                elif d.HasField("dim_param"):
                    dim.append(d.dim_param)
                else:
                # sys.exit("error: unknown dimension")
                    continue
            output_dims[output.name] = dim

    update_model_dims.update_inputs_outputs_dims(model, input_dims, output_dims)
    onnx.save(model, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hey launch')
    parser.add_argument('--onnx',type = str,default=None, help='task id')
    parser.add_argument('--save_path',type = list,default=None, help='task id')

    args,unknown = parser.parse_known_args()

    # TODO: 判断一下pt、onnx路径是否正确，shape格式是否对

    # TODO: 判断一下jit是否jit trace后的模型


    jit_export_onnx(args.onnx, args.save_path)