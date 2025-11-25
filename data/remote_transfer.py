import multiprocessing as mp

import paramiko


class RemoteTransfer:
    def __init__(self, hostname, port, username, password=None, key_filepath=None):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if key_filepath:
            self.ssh.connect(hostname, port, username=username, key_filename=key_filepath)
        else:
            self.ssh.connect(hostname, port, username=username, password=password)
        self.sftp = self.ssh.open_sftp()
        self.queue = mp.Queue()
        self.process = mp.Process(target=self._upload_worker, args=(self.queue,))
        self.process.start()

    def _upload_worker(self, queue):
        while True:
            task = queue.get()
            if task is None:
                break
            local_file, remote_file = task
            self._upload_file(local_file, remote_file)
            print(f"[RemoteTransfer]: Uploaded {local_file} to {remote_file}")

    def list_remote_dir(self, remote_dir):
        return self.sftp.listdir(remote_dir)
    
    def upload_file(self, local_file, remote_file):
        self.queue.put((local_file, remote_file))

    def _upload_file(self, local_file, remote_file):
        remote_tmp = remote_file + '.tmp'
        self.sftp.put(local_file, remote_tmp)
        self.sftp.rename(remote_tmp, remote_file)

    def download_file(self, remote_file, local_file):
        self.sftp.get(remote_file, local_file)

    def close(self):
        self.sftp.close()
        self.ssh.close()
