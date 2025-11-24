# Automated Journalist App

Includes the code-base for the NLP Lab Course for Automated Journalist App SOSE2024

## API Documentation

You can find the api (swagger) documentation under the link: http://127.0.0.1:8787/

## Installation:

You may create a Python environment either by pipenv or virtualenv. In
the following both installation steps are written. Please select either pipenv
or virtualenv and install it using one of those.

### pipenv
```
$ pipenv shell
$ pipenv install
```

### virtualenv
```
$ virtualenv -p python3 venv
$ source venv/bin/activate
```
My version:
```
$ python3.11 -m venv venv
$ source venv/bin/activate
```

- Install the app and its dependencies with pip. Inside the app root folder (the one containing `requirements.txt`, run the following command)

```
$ pip3 install -r requirements.txt
```
My version
```
$ pip install -r requirements.txt
```
```
ollama pull mistral
```

- Install Redis for caching
```
$ brew install redis
```

- Export the environment variables in the `.env` file.

```
$ set -a; source .env; set +a
```

## Run the Redis server for caching
```
$ brew services start redis
```


## Run the uvicorn server to deploy:
For development purposes, you may also start the with the "flask run" commmand.
If you run the app with "flask run", you can also use a Python debugger. For the
deployment server, use the following "uvicorn" command:

```
$ uvicorn app:asgi_app --port 8787 --host 127.0.0.1
```

## An example config.json file
In a file called config.json the env variables such as telegram, reddit api keys are stored.
You can find an example config file in the repo with name 'config-example.json'