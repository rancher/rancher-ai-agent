# Define target platforms, image builder and the fully qualified image name.
TARGET_PLATFORMS ?= linux/amd64,linux/arm64

REPO ?= rancher
DIRTY := $(shell if [ -n "$$(git status --porcelain --untracked-files=no)" ]; then echo "-dirty"; fi)
COMMIT ?= $(shell git rev-parse --short HEAD)
GIT_TAG ?= $(shell git tag -l --contains HEAD | head -n 1)

ifeq ($(DIRTY),)
ifneq ($(GIT_TAG),)
VERSION ?= $(GIT_TAG)
endif
endif
VERSION ?= 0.0.0-$(COMMIT)$(DIRTY)

TAG ?= $(VERSION)
IMAGE = $(REPO)/rancher-ai-agent:$(TAG)

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  build-image   Build production image locally"
	@echo "  push-image    Build and push multi-arch production image"
	@echo "  test          Run unit tests in container"
	@echo "  help          Show this help"

build-image:
	docker buildx build \
		--file package/Dockerfile \
		--platform=${TARGET_PLATFORMS} \
		-t ${IMAGE} \
		--load \
		.

push-image:
	docker buildx build \
		${IID_FILE_FLAG} \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT=$(COMMIT) \
		--file package/Dockerfile \
		--platform=${TARGET_PLATFORMS} \
		--sbom=true \
		--attest type=provenance,mode=max \
		-t ${IMAGE} \
		--push \
		.

test:
	docker buildx build \
		--file package/Dockerfile.test \
		--platform=${TARGET_PLATFORMS} \
		-t $(IMAGE)-test \
		--load \
		. && docker run --rm $(IMAGE)-test

.PHONY: help build-image push-image test
