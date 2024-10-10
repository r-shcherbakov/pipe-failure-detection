from clearml import Dataset

from settings import clearml_project, clearml_dataset_name, RAW_DATA_DIR

if __name__ == '__main__':
    try:
        old_version = [Dataset.get(
            dataset_project=clearml_project, 
            dataset_name=f"{clearml_dataset_name}"
            )]
    except ValueError as e:
        old_version = None
    dataset = Dataset.create(dataset_project=clearml_project,
                             dataset_name=f"{clearml_dataset_name}",
                             parent_datasets=old_version,
                             )  
    dataset.add_files(path=RAW_DATA_DIR)
    
    dataset.upload()    
    dataset.finalize()