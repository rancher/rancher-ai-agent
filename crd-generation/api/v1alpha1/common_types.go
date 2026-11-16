package v1alpha1

const (
	// DefaultNamespace is the namespace where AITaskDefinition and AITask objects live.
	DefaultNamespace = "cattle-ai-agent-system"

	// LabelTaskDefinition is set on every AITask with the name of its AITaskDefinition.
	LabelTaskDefinition = "ai.cattle.io/task-definition"
	// LabelTrigger is set on every AITask with the trigger ("scheduled" or "manual").
	LabelTrigger = "ai.cattle.io/trigger"
	// LabelTaskCredentials marks Secrets that may be referenced by credentialsRef.
	LabelTaskCredentials = "ai.cattle.io/task-credentials"
	// AnnotationScheduledAt holds the RFC3339 scheduled time of scheduled AITasks.
	AnnotationScheduledAt = "ai.cattle.io/scheduled-at"

	TriggerLabelScheduled = "scheduled"
	TriggerLabelManual    = "manual"

	// Condition types on AITaskDefinition.
	ConditionScheduleValid = "ScheduleValid" // set by the scheduler
	ConditionReady         = "Ready"         // aggregate

	// Condition reasons set by the scheduler.
	ReasonInvalidSchedule = "InvalidSchedule"
	ReasonUnknownTimeZone = "UnknownTimeZone"

	// Server-side apply field managers.
	FieldManagerScheduler = "ai-task-scheduler"
	FieldManagerExecutor  = "rancher-ai-agent"

	// MaxTaskDefinitionNameLength keeps "<name>-<unix minutes>" within 63 characters.
	MaxTaskDefinitionNameLength = 52
)

// TaskPhase is the lifecycle phase of an AITask.
// +kubebuilder:validation:Enum=Pending;Running;Succeeded;Failed
type TaskPhase string

const (
	TaskPhasePending   TaskPhase = "Pending" // created, not started yet
	TaskPhaseRunning   TaskPhase = "Running"
	TaskPhaseSucceeded TaskPhase = "Succeeded"
	TaskPhaseFailed    TaskPhase = "Failed"
)

// IsActive reports whether the phase counts as an active AITask.
// An empty phase is treated as Pending.
func (p TaskPhase) IsActive() bool {
	switch p {
	case "", TaskPhasePending, TaskPhaseRunning:
		return true
	}
	return false
}

// IsFinished reports whether the phase is terminal.
func (p TaskPhase) IsFinished() bool {
	switch p {
	case TaskPhaseSucceeded, TaskPhaseFailed:
		return true
	}
	return false
}

// TaskResult is the overall health verdict of a finished AITask.
// +kubebuilder:validation:Enum=Healthy;Warning;Critical
type TaskResult string

const (
	TaskResultHealthy  TaskResult = "Healthy"
	TaskResultWarning  TaskResult = "Warning"
	TaskResultCritical TaskResult = "Critical"
)

// CredentialsSecretRef identifies the key of a Secret (same namespace) holding a Rancher token.
type CredentialsSecretRef struct {
	// Name is the name of the Secret.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	Name string `json:"name"`

	// Key inside the Secret. Defaults to "token".
	// +optional
	// +kubebuilder:default="token"
	// +kubebuilder:validation:MaxLength=253
	Key string `json:"key,omitempty"`
}
