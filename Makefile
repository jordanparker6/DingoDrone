make drone:
	python3 dingo/main.py

make noiseprof:
	sox -c 1 ./dingo/assets/sample_background_noise.wav -n trim 0 2 noiseprof ./dingo/assets/sample_background_noise.noise-profile

make noisered:
	sox -c 1 voice_sample.wav output.wav noisered ./dingo/assets/sample_background_noise.noise-profile 0.5