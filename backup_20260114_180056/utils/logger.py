import os
import pickle
import tempfile
import time
import uuid
from copy import copy
from socket import gethostname

import numpy as np
import wandb


class WandBLogger(object):

    def __init__(self, config, variant):
        self.config = config

        if self.config.experiment_id is None:
            self.config.experiment_id = ""

        if self.config.prefix != "":
            self.config.project = "{}--{}".format(self.config.prefix, self.config.project)

        if self.config.output_dir == "":
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(self.config.output_dir, self.config.experiment_id)
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        try:
            os.environ["WANDB_API_KEY"] = config.wandb_config.wandb_api_key
            os.environ["WANDB_USER_EMAIL"] = config.wandb_config.wandb_user_email
            os.environ["WANDB_USERNAME"] = config.wandb_config.wandb_username
            os.environ["WANDB_MODE"] = "run"
        except:
            print(
                "Please set the wandb_config in the config file to use wandb logger, or turn off wandb logging."
            )

        self.run = wandb.init(
            reinit=True,
            config=self._variant,
            project=self.config.project,
            dir=self.config.output_dir,
            id=self.config.experiment_id + uuid.uuid4().hex,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode="online" if self.config.online else "offline",
            entity=self.config.entity,
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        with open(os.path.join(self.config.output_dir, filename), "wb") as fout:
            pickle.dump(obj, fout)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir
