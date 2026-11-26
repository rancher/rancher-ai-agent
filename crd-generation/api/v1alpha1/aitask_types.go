package v1alpha1

//go:generate controller-gen crd paths=./... output:crd:dir=../../crds
import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// AITaskSpec defines a single execution of an AITaskDefinition.
type AITaskSpec struct {
	// TaskDefinitionRef is the name of the AITaskDefinition (same namespace) this task executes.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=52
	TaskDefinitionRef string `json:"taskDefinitionRef"`

	// Prompt is the plain-language description of what Liz must do.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=8192
	Prompt string `json:"prompt"`

	// Agents lists the AIAgentConfig names this task may use. Empty means every enabled agent.
	// +optional
	// +listType=set
	// +kubebuilder:validation:MaxItems=64
	// +kubebuilder:validation:items:MinLength=1
	// +kubebuilder:validation:items:MaxLength=253
	Agents []string `json:"agents,omitempty"`

	// CredentialsRef is copied from the definition: the Secret with the Rancher token this task
	// runs with.
	// +optional
	CredentialsRef *CredentialsSecretRef `json:"credentialsRef,omitempty"`
}

// AITaskStatus defines the observed state of AITask.
type AITaskStatus struct {
	// +optional
	Phase TaskPhase `json:"phase,omitempty"`

	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty"`
	// +optional
	CompletionTime *metav1.Time `json:"completionTime,omitempty"`

	// Result is the overall health verdict of the task.
	// +optional
	Result TaskResult `json:"result,omitempty"`

	// PendingValidation is true while the run is paused (phase Running) waiting for a user to
	// answer a human validation request, in the follow-up chat or with the
	// ai.cattle.io/validation-response annotation.
	// +optional
	PendingValidation bool `json:"pendingValidation,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=ait,categories=ai
// +kubebuilder:printcolumn:name="Definition",type=string,JSONPath=`.spec.taskDefinitionRef`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Result",type=string,JSONPath=`.status.result`
// +kubebuilder:printcolumn:name="Validation",type=boolean,JSONPath=`.status.pendingValidation`
// +kubebuilder:printcolumn:name="Started",type=date,JSONPath=`.status.startTime`,priority=1
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// AITask is the Schema for the aitasks API: one execution of an AITaskDefinition.
type AITask struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// +kubebuilder:validation:Required
	// +kubebuilder:validation:XValidation:rule="self == oldSelf",message="spec is immutable"
	Spec   AITaskSpec   `json:"spec"`
	Status AITaskStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// AITaskList contains a list of AITask.
type AITaskList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []AITask `json:"items"`
}
