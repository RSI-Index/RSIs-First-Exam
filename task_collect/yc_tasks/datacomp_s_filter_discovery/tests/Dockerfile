FROM clip-datacomp-harbor:4a8df199

RUN rm -rf /app/project /app/output \
    && cp -a /opt/project /opt/candidate-clean \
    && mkdir -p /app/project /app/output /logs/verifier /tests
COPY policy.yaml baseline_contract.json policy_check.py artifact_check.py evaluate.py score.py /tests/
COPY --chmod=755 test.sh /tests/test.sh
RUN chmod -R a+rX /opt/candidate-clean /tests
CMD ["bash"]
