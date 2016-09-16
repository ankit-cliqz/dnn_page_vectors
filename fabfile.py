#!/usr/bin/python
# -*- coding: UTF-8 -*-

from fabric.api import task, run, local, put

@task
def upload_app():

    dst = '/ebs/code/dnn_page_vectors'

    run('rm -rf %s' % dst)
    local("rm -f app.tar.gz; git ls-files > list_of_files; " +
          "find extern -type f | grep -v /.git >> list_of_files; " +
          "tar -zcf app.tar.gz -T list_of_files; rm list_of_files")

    run('mkdir -p %s' % dst)
    put('app.tar.gz', '%s/app.tar.gz' % dst)
    run('cd %s/; tar -zxf %s/app.tar.gz' % (dst, dst))