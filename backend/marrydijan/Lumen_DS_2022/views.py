import shutil
import sys
from pathlib import Path

from django.core.exceptions import ValidationError
from django.shortcuts import render
import os.path
import os
from zipfile import ZipFile
sys.path.append(os.path.join("..", "..", "model3", "src"))
# from inference import main

# Create your views here.

from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseRedirect

DEFAULT_MODEL = "resnet50"

models = {
    "resnet50": None
}




def index(request):
    context = {
        "models": models.keys(),
        "has_multiple_models": len(models) > 1
    }
    return render(request, "Lumen_DS_2022/index.html", context)


def prepare_dirs(root_folder):
    if not os.path.isdir(root_folder):
        os.mkdir(root_folder)
    if not os.path.isdir(root_folder / "dataset"):
        os.mkdir(root_folder / "dataset")
    if not os.path.isdir(root_folder / "output_dir"):
        os.mkdir(root_folder / "output_dir")


def delete_dirs(root_folder):
    shutil.rmtree(root_folder)


valid_image_names = {"0.jpg", "90.jpg", "180.jpg", "270.jpg"}


def is_image_folder(path):
    return os.path.isdir(path) and set(os.listdir(path)) == valid_image_names


def create_target_csv(path):
    uuids = list(filter(is_image_folder, os.listdir(path / "data")))
    with open(path / "target.csv", 'w') as out_f:
        out_f.write("uuid\n")
        out_f.write('\n'.join(uuids))


def validate_target_csv(path):
    with open(path, 'r') as in_f:
        in_f.readline()
        if not all(is_image_folder(path.parent / "data" / fn.split(',')[0]) for fn in in_f):
            raise ValidationError("Some uuids don't have corresponding data folders")


def validate_dataset(dataset_root, desired_structure="standard", target_csv=None):
    if desired_structure == "standard":
        if not os.path.isdir(dataset_root / "data"):
            raise ValidationError("Missing data folder")
        if target_csv is not None and os.path.isfile(dataset_root / target_csv):
            validate_target_csv(target_csv)
        elif target_csv is not None:
            raise ValidationError(f"Desired target csv ({target_csv}) is not provided")
        else:
            create_target_csv(dataset_root)
            target_csv = "target.csv"
    return target_csv


def handle_uploaded_file(req_dir, uploaded_file, trainer_name=DEFAULT_MODEL):
    ext = uploaded_file.name[uploaded_file.name.rfind('.'):]
    uploaded_file_path = req_dir / ("uploaded_file" + ext)
    dataset_root = req_dir / "dataset"
    with open(uploaded_file_path, 'wb+') as out_f:
        for chunk in uploaded_file.chunks():
            out_f.write(chunk)

    if ext == ".zip":
        with ZipFile(uploaded_file_path, 'r') as zip_file:
            zip_file.extractall(dataset_root)
    target_csv = dataset_root / validate_dataset(dataset_root)
    output_dir = req_dir / "output_dir"
    parameters_path = "/mnt/1620E23720E21D8D/Dokumenti/Python/LumenDS2022/parameters_id-27.prms"
    main(
        [
            "--trainer_name", trainer_name,
            "--parameters_path", parameters_path,
            "--dataset_root", str(dataset_root),
            "--target_csv", str(target_csv),
            "--output_dir", str(output_dir),
            "--device", "cpu",
            "--batch_size", "1",
            "--num_workers", "4"
        ]
    )
    return output_dir




def upload(request):
    if "model_name" in request.POST:
        used_model = models[request.POST["model_name"]]
    else:
        used_model = models[DEFAULT_MODEL]
    if "file" not in request.FILES:
        return HttpResponseBadRequest("File not uploaded")

    # making a directory for each connection
    ip, port = request.environ["wsgi.input"].stream.raw._sock.getpeername()
    req_dir = Path(ip.replace(".", "_") + "-" + str(port))
    prepare_dirs(req_dir)
    output_dir = None
    try:
        output_dir = handle_uploaded_file(req_dir, request.FILES["file"], used_model)
    except ValidationError as e:
        return HttpResponseBadRequest(e.message)
    finally:
        delete_dirs(req_dir)
    with open(output_dir / "output.csv", 'r') as in_f:
        response = HttpResponse(in_f, mimetype="text/csv")
        response["Content-Disposition"] = "attachment; filename=out.csv"
        return response
    # return HttpResponseRedirect(request.META.get("HTTP_REFERER"))
