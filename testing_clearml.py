from clearml import Task

queue_number = 1

def main() -> None:
    task: Task = Task.init(project_name='dario/tests', task_name=f"testing the queue {queue_number}")

    task.set_base_docker(
        docker_image='rugg/aebias:latest',
        docker_arguments='--env LOCAL_PYTHON=/bin/python3.9 \
                          --env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 \
                          --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 \
                          --env CLEARML_AGENT_GIT_USER=ruggeri\
                          --env CLEARML_AGENT_GIT_PASS=glpat-bDqPbnKwZ98FrqdyUXgT\
                          --mount type=bind,source=/srv/nfs-data/ruggeri/datasets/IBD/,target=/data/ \
                          --mount type=bind,source=/srv/nfs-data/ruggeri/access_tokens/,target=/tokens/ \
                          --rm --ipc=host'.format(access_token=open("/tokens/gitlab_access_token.txt", "r").read())
    )

    task.execute_remotely(queue_name=f"rgai-gpu-01-2080ti:{queue_number}")

    print(" ---- task's main function runt ---- ")


if __name__ == "__main__":
    main()