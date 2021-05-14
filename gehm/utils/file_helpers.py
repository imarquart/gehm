import logging
from os import mkdir, getcwd
from os.path import exists, normpath
from pathlib import Path


def check_create_folder(folders,create_folder=True):
    """
    Checks and creates folder if necessary
    If path relative, checks also from current working directory
    Parameters
    ----------
    create_folder
    folder: String
        Folder

    Returns
    -------
    folder directory on success

    """
    folder_return_list=[]
    if isinstance(folders,list):
        for folder in folders:
            folder = Path(folder)

            # check if file
            if not folder.suffix == '':
                filename=folder.stem+folder.suffix
                db_folder=str(folder.parents[0])
            else:
                filename=""
                db_folder=str(folder)

            if not folder.is_absolute():
                if not (db_folder[0] in ('/','\\')):
                    db_folder= "/"+db_folder
                db_folder=getcwd() + db_folder

            if not exists(db_folder):
                if create_folder:
                    try:
                        logging.getLogger("Dirs Creator").info("Folder {} does not exist. Creating folder.".format(db_folder))
                        Path(db_folder).mkdir(parents=True)
                    except:
                        msg="Could not create folder {}".format(db_folder)
                        logging.getLogger("Dirs Creator").error(msg)
                        raise AttributeError(msg)
                    folder_return_list.append(normpath(db_folder+"/"+filename))
                else:
                        msg="Folder {} does not exist".format(db_folder)
                        logging.getLogger("Dirs Creator").error(msg)
                        raise AttributeError(msg)
                        return False
            else:
                folder_return_list.append(normpath(db_folder+"/"+filename))
    else:
        folders=[folders]
        folder_return_list= check_create_folder(folders, create_folder=create_folder)[0]
    
    return folder_return_list
