pipeline {
agent any

environment{
    PYTHON_VERSION = '3.12'
    VENV_NAME = 'venv'
}

stages
{
stage('Setup Python Environment'){
    steps{
        script{
            bat """
            python -m VENV ${VENV_NAME}
            .\\${VENV_NAME}
            \\Scripts\\activate.bat
            python -m pip install --upgrade pip
            pip install -r requirements.txt
            """

        }
    }
}
stage('Run Tests'){
    steps{
        script{
            bat """
            .\\${VENV_NAME}\\Scripts\\activate.bat
            pytest tests/
            """
        }
    }
}
stage('Build Model Artifacts'){
    steps{
        script{
            bat """
            .\\${VENV_NAME}\\Scripts\\activate.bat
            python src/train_model.py
            """
        }
    }
}
stage('Deploy Streamli app'){
    steps{
        script{
            bat """
           .\\${VENV_NAME}\\Scripts\\activate.bat
            streamlit run src/streamlit_app.py
            """
        }
    }
}

}

post{
    always{
        cleanWs()
    }
    success{
        echp 'Pipeline completed succesfully'
    }
    failure{
        echo 'Pipeline failed , Check the logs for details'
    }
}

}