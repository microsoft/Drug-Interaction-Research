# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/bin/bash

set -e

wget https://msracb.blob.core.windows.net/pub/igt-ckpt.zip
unzip igt-ckpt.zip
rm igt-ckpt.zip

wget https://msracb.blob.core.windows.net/pub/igt-data.zip
unzip igt-data.zip
rm igt-data.zip
