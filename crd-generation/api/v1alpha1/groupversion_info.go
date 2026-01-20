// +kubebuilder:object:generate=true
// +groupName=ai.cattle.io
package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	// GroupVersion is group version used to register these objects
	GroupVersion = schema.GroupVersion{Group: "ai.cattle.io", Version: "v1alpha1"}
)
