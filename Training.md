# Training on TPU 

* Do not start them by hand, as setting the connection to a Compute Engine by Hand is tricky and a waste of time. Instead:

`$ctpu up --zone=europe-west4-a  --tf-version=1.15 -preemptible --name=tpu-testv13 --tpu-size=v3-8 (In the overall Google Cloud Shell @ Project Quickpiq)`

 

* Dont use "--require-permissons" there seems to be a bug ("error adding the TPU's service account to the project's access control lists:")

* The Compute Engine is automatically connected with the TPU. No Environmental Variables have to be set (IF RUN IN CLOUD SHELL!!!!): 

`$ tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver()` 
(This is all that is needed inside Python to connect to the TPU Cluster)

# Print all available Devices:

`resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)`

This is the TPU initialization code that has to be at the beginning.
It can vary dependent on the code. For electra the TPU_Name has to be passed as the single variable (through the Command Line Parameters)

`tf.tpu.experimental.initialize_tpu_system(resolver)
#print("All devices: ", tf.config.list_logical_devices('TPU'))`

Either simple shut down TPU and! VM (HDD still billed) or delete both with ctpu command:

`$ ctpu delete --zone='europe-west4-a' --name=tpu-testv11`

The TPU can only use datasets from a Storage Bucket (gs://xxxx) 
Analyze Usage:

`capture_tpu_profile --tpu=$TPU_NAME  --monitoring_level=2`

start tensorboard directly in Cloud Shell (https://cloud.google.com/tpu/docs/tensorboard-setup?hl=de#streaming-trace-viewer)