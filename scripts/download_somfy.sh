#!/bin/bash
set -e
BASE="https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-Honeypot-Capture-7-1"
OUT="data/raw_iot23_downloads/benign_SomfyDoorlock"
mkdir -p "$OUT"

curl -L -o "$OUT/somfy-01.pcap" "$BASE/Somfy-01/2019-07-03-15-15-47-first_start_somfy_gateway.pcap"
curl -L -o "$OUT/somfy-02.pcap" "$BASE/Somfy-02/2019-07-03-16-41-09-192.168.1.158.pcap"
curl -L -o "$OUT/somfy-03.pcap" "$BASE/Somfy-03/2019-07-04-16-41-10-192.168.1.158.pcap"
curl -L -o "$OUT/somfy-04.pcap" "$BASE/Somfy-04/2019-07-05-16-41-14-192.168.1.158.pcap"
curl -L -o "$OUT/somfy-05.pcap" "$BASE/Somfy-05/2019-07-06-16-41-17-192.168.1.158.pcap"
curl -L -o "$OUT/somfy-06.pcap" "$BASE/Somfy-06/2019-07-07-16-41-19-192.168.1.158.pcap"

echo "SOMFY_DOWNLOAD_DONE"
