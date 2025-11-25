from data.remote_transfer import RemoteTransfer

host_name = "120.48.58.215"
port = 603
user_name = "root"
key_filepath = "/home/cyx/.ssh/id_rsa"
remote_transfer = RemoteTransfer(host_name, port, user_name, key_filepath=key_filepath)
# test listdir
list_dir = remote_transfer.list_remote_dir("/mnt/pfs/datasets")

# test download
# remote_transfer.download_folder("/mnt/pfs/datasets", "./temp/", overwrite=True)
remote_transfer.start_upload_process()
# remote_transfer.upload_file("./tests/upload_file.txt", "/mnt/pfs/upload_file.txt", overwrite=True)
# remote_transfer.upload_folder("./tests/", "/mnt/pfs/upload_tests/", overwrite=False)
remote_transfer.close()


    








