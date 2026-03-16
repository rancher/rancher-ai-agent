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
