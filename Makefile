train:
	modal run ops/train.py 

deploy:
	modal deploy ops/inference.py

test:
	INFERENCE_ENDPOINT_URL='https://pcubasm1--audio-cnn-inference-audioclassifier-inference.modal.run' pytest tests/test_inference_endpoint.py -v
