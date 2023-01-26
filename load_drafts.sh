echo "  ---> Loading AE"
/bin/python3.9 /workspaces/metamorphic-testing/auto_encoder.py
echo "  ---> Loading Classifier"
/bin/python3.9 /workspaces/metamorphic-testing/classifier.py
echo "  ---> Loading Aggragator"
/bin/python3.9 /workspaces/metamorphic-testing/pipeline_aggregator.py