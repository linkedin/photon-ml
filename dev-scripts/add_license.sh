#!/bin/bash
# Copyright 2017 LinkedIn Corp. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

# This script provides ability of applying/deapplying the LICENSE txt to all
# source files.

function apply_license {
    license="${1}" # license file
    shift
    files=("${@}") # array of files
    for i in ${files[@]};
    do
        if (( $(grep -c "www.apache.org/licenses" $i) == 0 )) # only apply license if a license does not already exist in the file
        then
            echo "Adding license for file: $i"
            cat ${license} $i > $i.licensed && mv $i.licensed $i; # no restriction for package name to be the first line in the file
        fi
    done
}

java_files=(`find . -regex '.*\.java\|.*\.gradle\|.*\.scala\|.*\.groovy' -type f`)
java_license=.java_apache_license.txt
rm -f $java_license
# Surround License with Java comment style
sed -e 's/^/ * /' -e 's/ $//' -e '$a\ */' -e '1i /*' dev-scripts/license_template.txt > $java_license

pyRsh_files=(`find . -regex '.*\.py\|.*\.sh\|.*\.R' -type f`)
pyRsh_license=.pyRsh_apache_license.txt
rm -f $pyRsh_license
# Surround License with Python/R comment style
sed -e 's/^/# /' -e 's/ $//' dev-scripts/license_template.txt > $pyRsh_license

# Apply license to java files
apply_license ${java_license} ${java_files[@]}

# Apply license to python, R and shell scripts
apply_license ${pyRsh_license} ${pyRsh_files[@]}

# Clean up
rm -f $java_license
rm -f $pyRsh_license