#!/bin/sh
set -e

# The paths docker-compose mounts volumes over are managed by Docker, not by
# this image. A named volume keeps whatever ownership it had at creation
# through every later image rebuild -- the Dockerfile's own chown only ever
# touches this image's layers, never a volume that already existed before it
# was mounted. That is exactly what happened to media_volume here: created
# once, apparently before some earlier permissions fix, it stayed root-owned
# across every rebuild since, and www-data (running as a static Dockerfile
# USER) could read it but never write to it.
#
# Fixed by starting the container as root, fixing ownership on every boot --
# cheap and idempotent, a chown on an already-correct tree does nothing -- and
# only then dropping to www-data before running anything that touches request
# data. USER is deliberately not set in the Dockerfile any more; this script
# is the thing that decides who the app actually runs as.
#
# mkdir first: nothing guarantees this path exists before the chown runs.
# docker-compose always mounts a volume here, but a bare `docker run` with no
# volume at all -- exactly what this project's own CI does to test that the
# image boots standalone -- leaves it missing, and `chown` on a path that
# does not exist kills the entrypoint under `set -e` before uvicorn ever
# starts. Caught by running that exact scenario after writing this script,
# not assumed from reading it.
mkdir -p /var/www/static
chown -R www-data:www-data /var/www/static

exec gosu www-data "$@"
