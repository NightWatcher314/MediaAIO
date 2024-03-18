import subprocess
import hashlib


class ProcessManager:
    _instance = None
    _dict = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def exec(self, command: str, token):
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        hash_object = hashlib.sha256()
        hash_object.update((command + token).encode("utf-8"))
        command_hash = hash_object.hexdigest()
        self._dict[command_hash] = process
        return command_hash

    def print_process_stdout_and_wait(self, process):
        while process.poll() is None:
            for line in process.stdout:
                print(line.decode("utf-8"))

    def get_process_by_hash(self, command_hash) -> subprocess.Popen:
        return self._dict[command_hash]

    def delete_process_by_hash(self, command_hash):
        del self._dict[command_hash]
