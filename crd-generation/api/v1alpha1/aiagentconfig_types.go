package v1alpha1

//go:generate controller-gen crd paths=./... output:crd:dir=../../../chart/agent/templates/crds
import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// AIAgentConfigSpec defines the desired state of AIAgentConfig
type AIAgentConfigSpec struct {
	// Name is the name of the AI agent
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// Description provides details about the agent's purpose
	// +optional
	Description string `json:"description,omitempty"`

	// SystemPrompt is the initial system-level instructions for the agent
	// +optional
	SystemPrompt string `json:"systemPrompt,omitempty"`

	// MCPURL is the Model Context Protocol server URL
	// +optional
	MCPURL string `json:"mcpURL,omitempty"`

	// AuthenticationType specifies the authentication method
	// +kubebuilder:validation:Enum=RANCHER;NONE;BASIC
	// +optional
	AuthenticationType string `json:"authenticationType,omitempty"`

	// AuthenticationSecret specifies the authentication secret
	// +optional
	AuthenticationSecret string `json:"authenticationSecret,omitempty"`

	// BuiltIn indicates if this is a built-in agent configuration
	// +optional
	BuiltIn bool `json:"builtIn,omitempty"`

	// Enabled indicates if this agent configuration is enabled
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// HumanValidationTools lists tools requiring human validation
	// +optional
	HumanValidationTools []HumanValidationTool `json:"humanValidationTools,omitempty"`

	// ToolSet specifies a predefined set of tools for the agent
	// +optional
	ToolSet string `json:"toolSet,omitempty"`
}

// HumanValidationTool defines a tool that requires human confirmation
type HumanValidationTool struct {
	// Name of the tool
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// Type of validation (CREATE, UPDATE, DELETE)
	// +kubebuilder:validation:Enum=CREATE;UPDATE;DELETE
	// +kubebuilder:validation:Required
	Type string `json:"type"`
}

// AIAgentConfigStatus defines the observed state of AIAgentConfig
type AIAgentConfigStatus struct {
	// Conditions represent the latest available observations of the AIAgentConfig's state
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// Phase represents the current phase of the agent configuration
	// +optional
	Phase string `json:"phase,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=aiac
// +kubebuilder:printcolumn:name="Name",type=string,JSONPath=`.spec.name`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// AIAgentConfig is the Schema for the aiagentconfigs API
type AIAgentConfig struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   AIAgentConfigSpec   `json:"spec,omitempty"`
	Status AIAgentConfigStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// AIAgentConfigList contains a list of AIAgentConfig
type AIAgentConfigList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []AIAgentConfig `json:"items"`
}
