from azureml.core import Workspace

#ws = Workspace.get(name='MLOps',
#                   subscription_id='c720cd74-e36b-4dc4-9780-6aa39b2e6524',
#                   resource_group='aml-resources')

for compute_name in ws.compute_targets:
    compute = ws.compute_targets[compute_name]
    print(compute.name, ":", compute.type)