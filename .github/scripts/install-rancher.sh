#!/bin/bash

# Helper script to run Rancher in a Docker container

VERSION="head"
CATTLE_SERVER_URL="https://172.17.0.1"
CATTLE_BOOTSTRAP_PASSWORD="asd"

if [ -n "$1" ]; then
  VERSION=$1
fi

if [ -n "$2" ]; then
  CATTLE_SERVER_URL="$2"
fi

if [ -n "$3" ]; then
  CATTLE_BOOTSTRAP_PASSWORD="$3"
fi

IMAGE="rancher/rancher:${VERSION}"

echo ""
echo "Starting '${IMAGE}' container ..."

echo ""
ID=$(docker run -d --restart=unless-stopped -p 80:80 -p 443:443 --privileged -e CATTLE_SERVER_URL=${CATTLE_SERVER_URL} -e CATTLE_BOOTSTRAP_PASSWORD=${CATTLE_BOOTSTRAP_PASSWORD} -e CATTLE_PASSWORD_MIN_LENGTH=3 ${IMAGE})

if [ $? -ne 0 ]; then
  echo "An error occurred running the Docker container"
  exit 1
fi

echo ""
echo "Container Id: ${ID}"

echo ""
echo "Waiting for backend to become ready"

TIME=0
while [[ "$(curl --insecure -s -m 5 -o /dev/null -w ''%{http_code}'' ${CATTLE_SERVER_URL})" != "200" ]]; do
  sleep 5;
  TIME=$((TIME + 5))
  echo "${TIME}s ..."
done

echo ""
echo "Login to Rancher"

RANCHER_TOKEN=$(curl -vkL \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"'"$CATTLE_BOOTSTRAP_PASSWORD"'" ,"responseType":"cookie"}' \
  -X POST \
  ${CATTLE_SERVER_URL}/v3-public/localProviders/local?action=login 2>&1 | \
  awk -F'[=;]' '/Set-Cookie/{gsub(" ","",$2);print $2;exit}')

echo ""
echo "Get kubeconfig from local cluster"

HEADER="Cookie: R_SESS=$RANCHER_TOKEN"
curl -kL \
  -H 'Content-Type: application/json' \
  -X POST \
  -H "$HEADER" \
  ${CATTLE_SERVER_URL}/v3/clusters/local?action=generateKubeconfig | \
  yq '.config' > kubeconfig.yaml

sleep 2;

echo ""
kubectl --kubeconfig=kubeconfig.yaml cluster-info
if [ $? -ne 0 ]; then
  echo "Unable to get Rancher resources"
  exit 1
fi

echo "Done"