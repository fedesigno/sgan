docker login eidos-service.di.unito.it
docker rmi eidos-service.di.unito.it/signoretta/sgan:eval
docker build -t eidos-service.di.unito.it/signoretta/sgan:eval . -f Dockerfile
docker push eidos-service.di.unito.it/signoretta/sgan:eval

docker service rm signoretta-sgan-eval
submit eidos-service.di.unito.it/signoretta/sgan:eval evaluate_model.py \
  --checkpoint_fold '/scratch/sgan/src/checkpoint/' \
  --model_path 'baseline'

docker service logs -f signoretta-sgan-eval