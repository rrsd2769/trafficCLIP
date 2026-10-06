#!/bin/bash
set -e
BASE="https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios"
OUT="data/raw_iot23_downloads"

curl -L -o "$OUT/benign_AmazonEcho/2018-09-21-capture.pcap" \
  "$BASE/CTU-Honeypot-Capture-5-1/2018-09-21-capture.pcap"

curl -L -o "$OUT/benign_PhilipsHue/2018-10-25-14-06-32-192.168.1.132.pcap.xz" \
  "$BASE/CTU-Honeypot-Capture-4-1/2018-10-25-14-06-32-192.168.1.132.pcap.xz"

bash scripts/download_somfy.sh

curl -L -o "$OUT/malware_Mirai/2019-01-10-21-06-26-192.168.1.199.pcap" \
  "$BASE/CTU-IoT-Malware-Capture-44-1/2019-01-10-21-06-26-192.168.1.199.pcap"

curl -L -o "$OUT/malware_HideAndSeek/2018-05-09-192.168.100.103.pcap" \
  "$BASE/CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.pcap"

curl -L -o "$OUT/malware_Muhstik/2018-05-21_capture.pcap" \
  "$BASE/CTU-IoT-Malware-Capture-3-1/2018-05-21_capture.pcap"

curl -L -o "$OUT/malware_Torii/2018-10-02-13-12-30-192.168.100.103.pcap" \
  "$BASE/CTU-IoT-Malware-Capture-20-1/2018-10-02-13-12-30-192.168.100.103.pcap"

curl -L -o "$OUT/malware_Torii/2018-10-03-15-22-32-192.168.100.113.pcap" \
  "$BASE/CTU-IoT-Malware-Capture-21-1/2018-10-03-15-22-32-192.168.100.113.pcap"

curl -L -o "$OUT/malware_Trojan/2019-01-10-14-34-38-192.168.1.197.pcap" \
  "$BASE/CTU-IoT-Malware-Capture-42-1/2019-01-10-14-34-38-192.168.1.197.pcap"

curl -L -o "$OUT/malware_Hakai/2018-07-31-15-15-09-192.168.100.113.pcap" \
  "$BASE/CTU-IoT-Malware-Capture-8-1/2018-07-31-15-15-09-192.168.100.113.pcap"

echo "IOT23_DOWNLOAD_DONE"
