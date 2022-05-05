import shutil
import sys
from pathlib import Path

from django.core.exceptions import ValidationError
from django.shortcuts import render
import os.path
import os
from zipfile import ZipFile
sys.path.append(os.path.join("..", "..", "model3", "src"))
from inference import main

# Create your views here.

from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseRedirect


def load_inference_args(path="inference_args.txt"):
    result = {}
    with open(path, 'r') as in_f:
        for line in in_f:
            if line.startswith('#'):
                continue
            arg_name, arg_val = line.strip().split()
            result[arg_name.lstrip('-')] = arg_val
    return result


DEFAULT_INF_ARGS = load_inference_args()


def index(request):
    models = {}
    context = {
        "models": models.keys(),
        "has_multiple_models": len(models) > 1
    }
    return render(request, "Lumen_DS_2022/index.html", context)


def prepare_dirs(*dirs):
    temp_dirs = []
    for directory in dirs:
        if not directory.is_dir():
            directory.mkdir()
            temp_dirs.append(directory)
    return temp_dirs


def delete_dirs(*dirs):  # must be topologically sorted
    for directory in dirs[::-1]:
        shutil.rmtree(directory)


valid_image_names = {"0.jpg", "90.jpg", "180.jpg", "270.jpg"}


def is_image_folder(path):
    return os.path.isdir(path) and set(os.listdir(path)) == valid_image_names


def create_target_csv(path):
    uuids = list(filter(lambda fn: is_image_folder(path / "data" / fn),
                        os.listdir(path / "data")))
    with open(path / "target.csv", 'w') as out_f:
        out_f.write("uuid\n")
        out_f.write('\n'.join(uuids))


def validate_target_csv(path):
    with open(path, 'r') as in_f:
        in_f.readline()
        if not all(is_image_folder(path.parent / "data" / fn.split(',')[0]) for fn in in_f):
            raise ValidationError("Some uuids don't have corresponding data folders")


def validate_dataset(dataset_root, target_csv=None, structure="standard"):
    if structure == "standard":
        if not (dataset_root / "data").is_dir():
            raise ValidationError("Missing data folder")
        if target_csv is not None and (dataset_root / target_csv).is_file():
            validate_target_csv(dataset_root / target_csv)
        elif target_csv is not None:
            raise ValidationError(f"Desired target csv ({target_csv}) is not provided")
        else:
            create_target_csv(dataset_root)
            target_csv = "target.csv"
    return target_csv


def handle_uploaded_file(req_dir, uploaded_file, dataset_root):
    ext = uploaded_file.name[uploaded_file.name.rfind('.'):]
    uploaded_file_path = req_dir / ("uploaded_file" + ext)
    with open(uploaded_file_path, 'wb+') as out_f:
        for chunk in uploaded_file.chunks():
            out_f.write(chunk)

    if ext == ".zip":
        with ZipFile(uploaded_file_path, 'r') as zip_file:
            zip_file.extractall(dataset_root)
    # os.remove(uploaded_file_path)


def run_inference(inf_args):
    inf_args["target_csv"] = \
        validate_dataset(inf_args["dataset_root"], inf_args["target_csv"])
    args_list = []
    for arg_name, arg_val in inf_args.items():
        args_list.append("--" + arg_name)
        args_list.append(str(arg_val))
    main(args_list)


def upload(request):
    # get default args (copied, to avoid modifying them)
    inf_args = DEFAULT_INF_ARGS.copy()
    # override the default args with sent headers
    for arg_name in filter(request.POST.__contains__, DEFAULT_INF_ARGS):
        inf_args[arg_name] = request.POST[arg_name]
    # ignore input paths in headers if input file is uploaded
    if "file" in request.FILES:
        # making a separate directory for each connection
        ip, port = request.environ["wsgi.input"].stream.raw._sock.getpeername()
        req_dir = Path(ip.replace(".", "_") + "-" + str(port))
        inf_args.update(
            dataset_root=req_dir / "dataset",
            output_dir=req_dir / "out",
            target_csv=request.POST.get("target_csv")
        )
    else:
        for it in ("dataset_root", "output_dir"):
            inf_args[it] = Path(inf_args[it])
        req_dir = inf_args["dataset_root"].parent
    temp_dirs = prepare_dirs(
        req_dir, inf_args["dataset_root"], inf_args["output_dir"]
    )
    try:
        if "file" in request.FILES:
            handle_uploaded_file(
                req_dir, request.FILES["file"], inf_args["dataset_root"]
            )
        run_inference(inf_args)
        with open(inf_args["output_dir"] / "output.csv", 'r') as in_f:
            return HttpResponse(in_f, headers={
                "Content-Type": "text/csv",
                "Content-Disposition": "attachment; filename=out.csv"
            })
    except ValidationError as e:
        return HttpResponseBadRequest(e.message)
    finally:
        delete_dirs(*temp_dirs)
