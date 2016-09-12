#!/usr/bin/env bash

scp -i ~/Documents/ankit_documents/ankit_info/ankit_ec2.pem -r * ubuntu@10.10.64.224:/ebs/download/code/cnn_dssm_v2/
