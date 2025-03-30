# 💻 Contributing: Join the Preprocessing Revolution! 🛠️

Eager to contribute? Great! We're excited to welcome new contributors to our project. Here's how you can get involved:

## 💡 New Ideas / Features Requests

If you wan't to request a new feature or you have detected an issue, please use the following link:
[ISSUES](https://github.com/piotrlaczkowski/keras-data-processor/issues)

## 🚀 Getting Started:

- [x] Fork the Repository: Visit our GitHub page, fork the repository, and clone it to your local machine.

- [x] Set Up Your Environment: Make sure you have TensorFlow, Loguru, and all necessary dependencies installed.

- [x] Make sure you have installed the pre-commit hook locally

  ??? installation-guide
  Before using pre-commit hook you need to install it in your python environment.

        ```bash
        conda install -c conda-forge pre-commit
        ```

        go to the root folder of this repository, activate your venv and use the following command:

        ```bash
        pre-commit install
        ```

- [x] Create a new branch to package your code

- [x] Use standarized commit message:

  `{LABEL}(KDP): {message}`

  This is very important for the automatic releases (semantic release) and to have clean history on the master branch.

  ??? Labels-types

        | Label    | Usage                                                                                                                                                                                                                                             |
        | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        | break    | `break` is used to identify changes related to old compatibility or functionality that breaks the current usage (major)                                                                                                                           |
        | feat     | `feat` is used to identify changes related to new backward-compatible abilities or functionality (minor)                                                                                                                                          |
        | init     | `init` is used to indentify the starting related to the project (minor)                                                                                                                                                                           |
        | enh      | `enh` is used to indentify changes related to amelioration of abilities or functionality (patch)                                                                                                                                                  |
        | build    | `build` (also known as `chore`) is used to identify **development** changes related to the build system (involving scripts, configurations, or tools) and package dependencies (patch)                                                            |
        | ci       | `ci` is used to identify **development** changes related to the continuous integration and deployment system - involving scripts, configurations, or tools (minor)                                                                                |
        | docs     | `docs`  is used to identify documentation changes related to the project; whether intended externally for the end-users or internally for the developers (patch)                                                                                  |
        | perf     | `perf`  is used to identify changes related to backward-compatible **performance improvements** (patch)                                                                                                                                           |
        | refactor | `refactor` is used to identify changes related to modifying the codebase, which neither adds a feature nor fixes a bug - such as removing redundant code, simplifying the code, renaming variables, etc.<br />i.e. handy for your wip ; ) (patch) |
        | style    | `style`  is used to identify **development** changes related to styling the codebase, regardless of the meaning - such as indentations, semi-colons, quotes, trailing commas, and so on (patch)                                                   |
        | test     | `test` is used to identify **development** changes related to tests - such as refactoring existing tests or adding new tests. (minor)                                                                                                             |
        | fix      | `fix`  is used to identify changes related to backward-compatible bug fixes. (patch)                                                                                                                                                              |
        | ops      | `ops` is used to identify changes related to deployment files like `values.yml`, `gateway.yml,` or `Jenkinsfile` in the **ops** directory. (minor)                                                                                                |
        | hotfix   | `hotfix` is used to identify **production** changes related to backward-compatible bug fixes (patch)                                                                                                                                              |
        | revert   | `revert` is used to identify backward changes (patch)                                                                                                                                                                                             |
        | maint    | `maint` is used to identify **maintenance** changes related to project (patch)                                                                                                                                                                    |

- [x] Create your first Merge Request (MR) as soon as possible.

  > Merge requests will be responsible for semantic-release storytelling and so use them wisely! The changelog report generated automatically will be based on your commits merged into main branch and should cover all the things you did for the project, as an example:

- [x] Separate your merge requests based on LABEL or functionality if you are working on `feat` label

  > This about what part of feature you are working on, (messages) i.e.:

        - `initializaing base pre-processing code`
        - `init repo structure`
        - `adding pre-processing unit-tests`

- [x] Once the code is ready create a Merge Request (MR) into the MAIN branch with a proper naming convention

  > The name of your MR should follow the same exact convention as your commits (we have a dedicated check for this in the CI):

        `{LABEL}(KDP): {message}`

- [x] Use small Merge Requests but do them more ofthen < 400 lines for quicker and simple review and not the whole project !

- [x] Ask for a Code Review !

- [x] Once your MR is approved, solve all your unresolved conversation and pass all the CI check before you can merge it.

- [x] All the Tests for your code should pass -> REMEMBER NO TESTS = NO MERGE 🚨
