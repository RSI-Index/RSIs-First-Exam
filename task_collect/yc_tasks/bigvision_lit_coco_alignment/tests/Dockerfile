FROM clip-bigvision-lit-harbor:8921d514

RUN rm -rf /app/project /app/output \
    && cp -a /opt/project /opt/candidate-clean \
    && mkdir -p /app/project /app/output /logs/verifier /tests
COPY policy.yaml baseline_contract.json policy_check.py artifact_check.py evaluate.py score.py /tests/
COPY --chmod=755 test.sh /tests/test.sh
RUN python3 -m pip install pyyaml==6.0.2 \
    && chmod -R a+rX /opt/candidate-clean /tests
CMD ["bash"]

