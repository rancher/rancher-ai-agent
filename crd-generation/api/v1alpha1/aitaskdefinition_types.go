package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// AITaskDefinitionSpec defines a recurring (or manual-only) task for Liz.
type AITaskDefinitionSpec struct {
	// DisplayName is the human readable name of the task.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	DisplayName string `json:"displayName"`

	// Prompt is the plain-language description of what Liz must do.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=8192
	Prompt string `json:"prompt"`

	// Schedule is a cron expression (5 fields) or macro (@hourly, @daily, @weekly, @monthly).
	// Empty means "manual only". CRON_TZ / TZ prefixes are not allowed; use TimeZone.
	// +optional
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:XValidation:rule="!self.contains('TZ=')",message="CRON_TZ / TZ prefixes are not allowed in schedule; use timeZone"
	Schedule string `json:"schedule,omitempty"`

	// TimeZone is an IANA time zone name, e.g. "Europe/Madrid". Defaults to UTC.
	// +optional
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=64
	TimeZone *string `json:"timeZone,omitempty"`

	// Suspend stops new scheduled AITasks from being created.
	// +optional
	Suspend bool `json:"suspend,omitempty"`

	// Agents lists the AIAgentConfig names the task may use. Empty means every eligible agent.
	// +optional
	// +listType=set
	// +kubebuilder:validation:MaxItems=64
	// +kubebuilder:validation:items:MinLength=1
	// +kubebuilder:validation:items:MaxLength=253
	Agents []string `json:"agents,omitempty"`

	// CredentialsRef references the Secret (same namespace) with the Rancher token the task runs
	// with. The Secret must have the label ai.cattle.io/task-credentials=true. Without it, runs
	// fail unless the RANCHER_API_TOKEN env var (testing only) is set.
	// +optional
	CredentialsRef *CredentialsSecretRef `json:"credentialsRef,omitempty"`
}

// AITaskDefinitionStatus defines the observed state of AITaskDefinition.
type AITaskDefinitionStatus struct {
	// Conditions: ScheduleValid (scheduler), Ready (aggregate).
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// LastScheduleTime is the last scheduled time that was handled. Owned by the scheduler.
	// +optional
	LastScheduleTime *metav1.Time `json:"lastScheduleTime,omitempty"`

	// NextRunTime is the next scheduled time.
	// +optional
	NextRunTime *metav1.Time `json:"nextRunTime,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=aitd,categories=ai
// +kubebuilder:printcolumn:name="Display Name",type=string,JSONPath=`.spec.displayName`
// +kubebuilder:printcolumn:name="Schedule",type=string,JSONPath=`.spec.schedule`
// +kubebuilder:printcolumn:name="Suspend",type=boolean,JSONPath=`.spec.suspend`
// +kubebuilder:printcolumn:name="Ready",type=string,JSONPath=`.status.conditions[?(@.type=="Ready")].status`
// +kubebuilder:printcolumn:name="Next Run",type=date,JSONPath=`.status.nextRunTime`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
// +kubebuilder:validation:XValidation:rule="size(self.metadata.name) <= 52",message="AITaskDefinition name must be no more than 52 characters"

// AITaskDefinition is the Schema for the aitaskdefinitions API.
type AITaskDefinition struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// +kubebuilder:validation:Required
	Spec   AITaskDefinitionSpec   `json:"spec"`
	Status AITaskDefinitionStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// AITaskDefinitionList contains a list of AITaskDefinition.
type AITaskDefinitionList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []AITaskDefinition `json:"items"`
}
