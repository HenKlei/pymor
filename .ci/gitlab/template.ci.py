#!/usr/bin/env python3

tpl = '''# THIS FILE IS AUTOGENERATED -- DO NOT EDIT #
#   Edit and Re-run .ci/gitlab/template.ci.py instead       #

stages:
  - sanity
  - test
  - build
  - install_checks
  - deploy

#************ definition of base jobs *********************************************************************************#

.test_base:
    retry:
        max: 2
        when:
            - runner_system_failure
            - stuck_or_timeout_failure
            - api_failure
    only: ['branches', 'tags', 'triggers', 'merge-requests']
    except:
        - /^staging/.*$/i
    variables:
        PYPI_MIRROR_TAG: {{pypi_mirror_tag}}

.pytest:
    extends: .test_base
    tags:
      - long execution time
    environment:
        name: unsafe
    stage: test
    after_script:
      - .ci/gitlab/after_script.bash
    artifacts:
        name: "$CI_JOB_STAGE-$CI_COMMIT_REF_SLUG"
        expire_in: 3 months
        paths:
            - src/pymortests/testdata/check_results/*/*_changed
            - coverage.xml
            - memory_usage.txt
        reports:
            junit: test_results.xml

{# note: only Vanilla and numpy runs generate coverage or test_results so we can skip others entirely here #}
.submit:
    extends: .test_base
    retry:
        max: 2
        when:
            - always
    environment:
        name: safe
    except:
        - /^github\/PR_.*$/
        - /^staging/.*$/i
    stage: deploy
    script: .ci/gitlab/submit.bash

.docker-in-docker:
    tags:
      - dustin
    extends: .test_base
    retry:
        max: 2
        when:
            - always
    {# this is intentionally NOT moving with CI_IMAGE_TAG #}
    image: pymor/docker-in-docker:d1b5ebb4dc42a77cae82411da2e503a88bb8fb3a
    variables:
        DOCKER_HOST: tcp://docker:2375/
        DOCKER_DRIVER: overlay2
    before_script:
        - 'export SHARED_PATH="${CI_PROJECT_DIR}/shared"'
        - mkdir -p ${SHARED_PATH}
    services:
        - docker:dind
    environment:
        name: unsafe


# this should ensure binderhubs can still build a runnable image from our repo
.binder:
    extends: .docker-in-docker
    stage: install_checks
    variables:
        IMAGE: ${CI_REGISTRY_IMAGE}/binder:${CI_COMMIT_REF_SLUG}
        CMD: "jupyter nbconvert --to notebook --execute /pymor/.ci/ci_dummy.ipynb"
        USER: juno

.wheel:
    extends: .docker-in-docker
    stage: build
    only: ['branches', 'tags', 'triggers']
    variables:
        TEST_OS: "{{ ' '.join(testos) }}"
    artifacts:
        paths:
        # cannot use exported var from env here
        - ${CI_PROJECT_DIR}/shared/pymor*manylinux*whl
        expire_in: 1 week


.check_wheel:
    extends: .test_base
    stage: install_checks
    services:
      - pymor/devpi:1
    dependencies:
    {%- for PY in pythons %}
    {%- for ML in manylinuxs %}
      - "wheel {{ML}} py{{PY[0]}} {{PY[2]}}"
    {%- endfor %}
    {%- endfor %}
    before_script:
      - pip3 install devpi-client
      - devpi use http://pymor__devpi:3141/root/public --set-cfg
      - devpi login root --password none
      - devpi upload --from-dir --formats=* ./shared
    only: ['branches', 'tags', 'triggers']
    # the docker service adressing fails on other runners
    tags: [mike]

#******** end definition of base jobs *********************************************************************************#

#******* sanity stage

# this step makes sure that on older python our install fails with
# a nice message ala "python too old" instead of "SyntaxError"
verify setup.py:
    extends: .test_base
    image: python:3.5-alpine
    stage: sanity
    script:
        - python setup.py egg_info

ci setup:
    extends: .docker-in-docker
    stage: sanity
    script: ./.ci/gitlab/ci_sanity_check.bash "{{ ' '.join(pythons) }}"


{%- for PY in pythons %}
{%- for ML in manylinuxs %}
wheel {{ML}} py{{PY[0]}} {{PY[2]}}:
    extends: .wheel
    variables:
        PYVER: "{{PY}}"
    script: bash .ci/gitlab/wheels.bash {{ML}}
{% endfor %}
{% endfor %}

{% for OS in testos %}
check_wheel {{loop.index}}:
    extends: .check_wheel
    image: pymor/deploy_checks:devpi_{{OS}}
    script: devpi install pymor[full]
{% endfor %}


# THIS FILE IS AUTOGENERATED -- DO NOT EDIT #
#   Edit and Re-run .ci/gitlab/template.ci.py instead       #

'''


import os
import jinja2
import sys
from itertools import product
from pathlib import Path  # python3 only
from dotenv import dotenv_values
tpl = jinja2.Template(tpl)
pythons = ['3.6', '3.7', '3.8']
oldest = [pythons[0]]
newest = [pythons[-1]]
test_scripts = [("mpi", pythons, 1), ("notebooks_dir", pythons, 1),  ("pip_installed", pythons, 1),
    ("vanilla", pythons, 1), ("numpy_git", newest, 1), ("oldest", oldest, 1),]
# these should be all instances in the federation
binder_urls = [f'https://{sub}.mybinder.org/build/gh/pymor/pymor' for sub in ('gke', 'turing', 'gesis')]
testos = ['centos_8', 'debian_buster', 'debian_testing']

env_path = Path(os.path.dirname(__file__)) / '..' / '..' / '.env'
env = dotenv_values(env_path)
ci_image_tag = env['CI_IMAGE_TAG']
pypi_mirror_tag = env['PYPI_MIRROR_TAG']
manylinuxs = [1, 2010, 2014]
with open(os.path.join(os.path.dirname(__file__), 'ci.yml'), 'wt') as yml:
    matrix = [(sc, py, pa) for sc, pythons, pa in test_scripts for py in pythons]
    yml.write(tpl.render(**locals()))
