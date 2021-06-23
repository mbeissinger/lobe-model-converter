"""
Create an OpenVINO model package to upload to Azure Blob Storage and use IoT Hub module update twin to update the Azure Percept AzureEyeModule.
"""
import argparse
import os
import json
import zipfile
import datetime
from azure.storage.blob import (
    BlockBlobService,
    BlobPermissions,
)
from azure.iot.hub import IoTHubRegistryManager
from azure.iot.hub.models import Twin, TwinProperties

def create_openvino_image_classification_model_config(model_filepath, label_filename='labels.txt'):
    """
    Create the AzureEyeModule config.json file for an image classification model. Returns the config filepath.
    """
    # Create the config.json file
    config = {
        "DomainType": "classification",
        "LabelFileName": label_filename,
        "ModelFileName": os.path.basename(model_filepath) # model filepath is the .xml openvino model file
    }
    # write the config.json file in the model directory
    config_filepath = os.path.join(os.path.dirname(model_filepath), "config.json")
    with open(config_filepath, "w") as f:
        json.dump(config, f)
    return config_filepath

def zip_openvino_image_classification_model_package(config_filepath):
    """
    Zip the model directory for uploading to IoT Hub. Return the zip filepath.
    """
    # read the config json
    with open(config_filepath, "r") as f:
        config = json.load(f)
    # create the zip file from config.json, the label file, and the model xml and bin files
    config_dirname = os.path.dirname(os.path.abspath(config_filepath))
    model_no_ext = os.path.splitext(config["ModelFileName"])[0]
    model_bin_filename = model_no_ext + ".bin"  # get the model .bin filename from the .xml file name
    # create the zip filepath from the model name
    zip_filepath = model_no_ext + ".zip"
    with zipfile.ZipFile(zip_filepath, "w") as zf:
        zf.write(config_filepath, arcname="config.json")
        zf.write(os.path.join(config_dirname, config["LabelFileName"]), arcname=config["LabelFileName"])
        zf.write(os.path.join(config_dirname, config["ModelFileName"]), arcname=config["ModelFileName"])
        zf.write(os.path.join(config_dirname, model_bin_filename), arcname=os.path.basename(model_bin_filename))
    return zip_filepath
    
def upload_model_zip(model_zip_filepath, model_container_name, storage_account_name, storage_account_key):
    """
    Upload the OpenVINO model package to Azure Blob Storage and return the download URL.
    """
    # create a BlockBlobService object with Azure storage account name and key
    block_blob_service = BlockBlobService(account_name=storage_account_name, account_key=storage_account_key)
    # create a container for the model
    block_blob_service.create_container(model_container_name, fail_on_exist=False)
    # upload the model package to the container
    model_blob_name = os.path.basename(model_zip_filepath)
    block_blob_service.create_blob_from_path(
        container_name=model_container_name,
        blob_name=model_blob_name,
        file_path=model_zip_filepath,
    )
    # get the model download URL
    model_download_url = block_blob_service.make_blob_url(
        model_container_name,
        model_blob_name,
        protocol='https',
        sas_token=block_blob_service.generate_blob_shared_access_signature(      
            container_name=model_container_name,
            blob_name=model_blob_name,
            permission=BlobPermissions.READ,
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        )
    )
    return model_download_url


def update_percept_module_twin(model_download_url, connection_string, device_id, module_id='azureeyemodule'):
    """
    Update the Azure IoT Hub module twin to use the new model download URL, which will cause the Percept kit to
    download and run the new model.

    connection_string, device_id come from IoT Hub:
    # Go to https://portal.azure.com
    # Select your IoT Hub
    # Click on Shared access policies
    # Click 'service' policy on the right (or another policy having 'service connect' permission)
    # Copy Connection string--primary key
    """
    iothub_registry_manager = IoTHubRegistryManager(connection_string)
    module_twin = iothub_registry_manager.get_module_twin(device_id, module_id)
    print (f"Module twin properties before update:\n{module_twin.properties}")

    # Update twin
    twin_patch = Twin()
    twin_patch.properties = TwinProperties(desired={"ModelZipUrl": model_download_url})
    updated_module_twin = iothub_registry_manager.update_module_twin(device_id, module_id, twin_patch, module_twin.etag)
    print (f"Module twin properties after update:\n{updated_module_twin.properties}")


if __name__ == '__main__':
    # Create a command line parser with the model filepath, Azure Storage account name, key, and model container name options
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True, help="Path to the OpenVINO model .xml file")
    parser.add_argument('--storage-account-name', type=str, required=True, help='Azure Storage account name')
    parser.add_argument('--storage-account-key', type=str, required=True, help='Azure Storage account key')
    parser.add_argument('--storage-container-name', type=str, required=True, help='Azure Storage model container name')
    parser.add_argument('--iothub-connection-string', type=str, required=True, help='IoT Hub connection string')
    parser.add_argument('--device-id', type=str, required=True, help='IoT Hub Percept device id')
    # Parse the command line arguments
    args = parser.parse_args()
    # Create the OpenVINO model package
    config_filepath = create_openvino_image_classification_model_config(args.model)
    # Zip the model package
    zip_filepath = zip_openvino_image_classification_model_package(config_filepath)
    # Upload the model package to Azure Storage
    model_download_url = upload_model_zip(zip_filepath, args.storage_container_name, args.storage_account_name, args.storage_account_key)
    # Update the Azure IoT Hub module twin to use the new model package version
    update_percept_module_twin(model_download_url, args.iothub_connection_string, args.device_id)
