{{- define "supplygraph.fullname" -}}
{{- if contains .Chart.Name .Release.Name }}{{ .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}{{ printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}{{ end }}
{{- end }}

{{- define "supplygraph.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end }}

{{- define "supplygraph.selector" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "supplygraph.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}
{{- end }}

{{- define "supplygraph.claim" -}}
{{ .Values.persistence.existingClaim | default (printf "%s-data" (include "supplygraph.fullname" .)) }}
{{- end }}

{{- define "supplygraph.tokenSecret" -}}
{{ .Values.api.token.existingSecret | default (printf "%s-api-token" (include "supplygraph.fullname" .)) }}
{{- end }}

{{- define "supplygraph.hasToken" -}}
{{- if or .Values.api.token.value .Values.api.token.existingSecret }}true{{ end }}
{{- end }}

{{/* Environment shared by the server and the pipeline. */}}
{{- define "supplygraph.env" -}}
- name: SUPPLYGRAPH_DATA_DIR
  value: /data
- name: SUPPLYGRAPH_ACTIONS_LOG
  value: /data/actions/events.jsonl
- name: SUPPLYGRAPH_LLM_MODEL
  value: {{ .Values.config.llmModel | quote }}
{{- with .Values.config.opensanctionsUrl }}
- name: OPENSANCTIONS_URL
  value: {{ . | quote }}
{{- end }}
{{- with .Values.neo4j.uri }}
- name: NEO4J_URI
  value: {{ . | quote }}
- name: NEO4J_USER
  value: {{ $.Values.neo4j.user | quote }}
{{- end }}
{{- end }}

{{- define "supplygraph.envFrom" -}}
{{- if or .Values.secrets.existingSecret .Values.secrets.values }}
envFrom:
  - secretRef:
      name: {{ .Values.secrets.existingSecret | default (printf "%s-env" (include "supplygraph.fullname" .)) }}
{{- end }}
{{- end }}
