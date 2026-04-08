package v1alpha1

//go:generate controller-gen crd paths=./... output:crd:dir=../../../chart/agent/templates/crds
import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
)

// UIToolDefinition defines a single UI tool
type UIToolDefinition struct {
	// Name of the UI tool
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// Category classifies the tool type (e.g., code, card, table)
	// +kubebuilder:validation:Required
	Category string `json:"category"`

	// Description explains what this UI tool does
	// +kubebuilder:validation:Required
	Description string `json:"description"`

	// Prompt is used by the LLM to understand when and how to select this tool
	// +kubebuilder:validation:Required
	Prompt string `json:"prompt"`

	// Enabled indicates whether this UI tool is active
	// +kubebuilder:validation:Optional
	// +optional
	Enabled *bool `json:"enabled,omitempty"`

	// Metadata describes tool capabilities and constraints
	// +kubebuilder:validation:Optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +optional
	Metadata *runtime.RawExtension `json:"metadata,omitempty"`

	// Schema defines the input validation schema for this tool (standard JSON Schema)
	// +kubebuilder:validation:Required
	// +kubebuilder:pruning:PreserveUnknownFields
	Schema runtime.RawExtension `json:"schema"`

	// Revision number of this UI tool definition (for change detection)
	// +kubebuilder:validation:Optional
	// +optional
	Revision *int32 `json:"revision,omitempty"`

	// Default values for user-editable tool fields
	// +kubebuilder:validation:Optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +optional
	DefaultValues *runtime.RawExtension `json:"defaultValues,omitempty"`
}

// UIToolsConfigConfig defines the spec-level configuration for UIToolsConfig
type UIToolsConfigConfig struct {
	// System prompt for the AI agent to use when deciding how to utilize the UI tools
	// +kubebuilder:validation:Optional
	// +optional
	SystemPrompt string `json:"systemPrompt,omitempty"`

	// Whether this UIToolsConfig is enabled and should be used by the agent
	// +kubebuilder:validation:Optional
	// +optional
	Enabled *bool `json:"enabled,omitempty"`

	// Revision number of this UIToolsConfig (for change detection)
	// +kubebuilder:validation:Optional
	// +optional
	Revision *int32 `json:"revision,omitempty"`

	// Maximum number of UI tools the LLM can select per response (0 = unlimited)
	// +kubebuilder:validation:Optional
	// +optional
	MaxTools *int32 `json:"maxTools,omitempty"`

	// Default values for user-editable config fields
	// +kubebuilder:validation:Optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +optional
	DefaultValues *runtime.RawExtension `json:"defaultValues,omitempty"`
}

// UIToolsConfigSpec defines the desired state of UIToolsConfig
type UIToolsConfigSpec struct {
	// Tools is the list of UI tool definitions
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinItems=1
	Tools []UIToolDefinition `json:"tools"`

	// Config contains spec-level configuration
	// +kubebuilder:validation:Required
	Config UIToolsConfigConfig `json:"config"`
}

// UIToolsConfigStatus defines the observed state of UIToolsConfig
type UIToolsConfigStatus struct {
	// Conditions represent the latest available observations of the UIToolsConfig's state
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// Phase represents the current phase of the UI tools configuration
	// +optional
	Phase string `json:"phase,omitempty"`

	// ToolCount is the number of tools currently loaded
	// +optional
	ToolCount *int32 `json:"toolCount,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=uits
// +kubebuilder:printcolumn:name="Tool Count",type=integer,JSONPath=`.spec.tools | length`
// +kubebuilder:printcolumn:name="Enabled",type=boolean,JSONPath=`.spec.config.enabled`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// UIToolsConfig is the Schema for the uitoolsconfigs API
type UIToolsConfig struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   UIToolsConfigSpec   `json:"spec,omitempty"`
	Status UIToolsConfigStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// UIToolsConfigList contains a list of UIToolsConfig
type UIToolsConfigList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []UIToolsConfig `json:"items"`
}
