"""
chat_assistant.py
AI-powered chat assistant for explaining pipeline results.
Users can select specific flows, findings, or decisions and ask questions.
"""

import os
import json
from openai import OpenAI


class ResultsChatAssistant:
    """
    Chat assistant that answers questions about pipeline results.
    Context-aware - knows about the specific result being analyzed.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history = []
    
    def analyze_flow(self, flow_data: dict, question: str) -> str:
        """
        Answer questions about a specific classified flow.
        
        Parameters
        ----------
        flow_data : dict
            Contains: attack_type, confidence, class_probs, shap_values, features
        question : str
            User's question
        
        Returns
        -------
        str
            AI's explanation
        """
        system_prompt = f"""You are an expert cybersecurity analyst explaining network flow analysis results.

The user is asking about a specific network flow with these details:
- Classification: {flow_data.get('attack_type', 'Unknown')}
- Confidence: {flow_data.get('confidence', 0):.2%}
- Top predicted classes: {json.dumps(flow_data.get('class_probs', {}), indent=2)}

SHAP Feature Importance (top contributing features):
{self._format_shap_values(flow_data.get('shap_values', {}))}

Your job:
1. Explain WHY this flow was classified this way
2. Interpret SHAP values in plain English
3. Highlight suspicious patterns
4. Be specific and educational

Keep responses concise (3-5 sentences) unless asked for details.
"""
        
        return self._get_response(system_prompt, question)
    
    def analyze_finding(self, finding_data: dict, question: str) -> str:
        """
        Answer questions about a threat hunting finding.
        
        Parameters
        ----------
        finding_data : dict
            Contains: rule_id, rule_name, attack_type, severity, description, evidence
        question : str
            User's question
        
        Returns
        -------
        str
            AI's explanation
        """
        system_prompt = f"""You are an expert SOC analyst explaining threat hunting findings.

The user is asking about this threat hunting finding:
- Rule: {finding_data.get('rule_id', '')} - {finding_data.get('rule_name', '')}
- Attack Type: {finding_data.get('attack_type', 'Unknown')}
- Severity: {finding_data.get('severity', 'MEDIUM')}
- Description: {finding_data.get('description', '')}
- Affected Flows: {finding_data.get('affected_flows', 0)}

Evidence:
{json.dumps(finding_data.get('evidence', {}), indent=2)}

Your job:
1. Explain what this finding means
2. Why it's concerning
3. What the attacker might be trying to do
4. Whether the severity is appropriate

Be clear and actionable.
"""
        
        return self._get_response(system_prompt, question)
    
    def analyze_decision(self, decision_data: dict, question: str) -> str:
        """
        Answer questions about an AI agent decision.
        
        Parameters
        ----------
        decision_data : dict
            Contains: action, reasoning, severity, priority, signature_rule, escalate_to_human
        question : str
            User's question
        
        Returns
        -------
        str
            AI's explanation
        """
        system_prompt = f"""You are an expert AI security analyst explaining autonomous response decisions.

The user is asking about this AI agent decision:
- Action Taken: {decision_data.get('action', 'UNKNOWN')}
- Attack Type: {decision_data.get('attack_type', '')}
- Severity: {decision_data.get('severity', 'MEDIUM')}
- Priority: {decision_data.get('priority', 3)}/5
- Escalate to Human: {decision_data.get('escalate_to_human', False)}

Agent's Reasoning:
"{decision_data.get('reasoning', 'No reasoning provided')}"

Signature Rule Created:
{decision_data.get('signature_rule', 'None')}

Your job:
1. Explain WHY the agent made this decision
2. Whether it's appropriate
3. What alternatives existed
4. The trade-offs involved

Be analytical and educational.
"""
        
        return self._get_response(system_prompt, question)
    
    def general_question(self, context: dict, question: str) -> str:
        """
        Answer general questions about the pipeline or results.
        
        Parameters
        ----------
        context : dict
            Pipeline run context (metrics, counts, etc.)
        question : str
            User's question
        
        Returns
        -------
        str
            AI's explanation
        """
        system_prompt = f"""You are an expert in autonomous threat hunting and ML-based security systems.

Pipeline Context:
- Records Processed: {context.get('records_processed', 0):,}
- Attacks Detected: {context.get('attacks_detected', 0):,}
- Detection Rate: {context.get('detection_rate', 0):.1%}
- Threat Findings: {context.get('findings_count', 0)}
- AI Decisions Made: {context.get('decisions_count', 0)}

Dataset: CIC-DDoS2019 (18 attack types, 30 features)
Model: Random Forest with SHAP explainability
Agent: GPT-4 for autonomous decision-making

Answer questions about:
- How the system works
- What metrics mean
- How to interpret results
- Best practices in threat hunting

Be clear and educational.
"""
        
        return self._get_response(system_prompt, question)
    
    def _get_response(self, system_prompt: str, user_question: str) -> str:
        """Get response from GPT with conversation history."""
        try:
            # Add to conversation history
            if not self.conversation_history:
                self.conversation_history.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            self.conversation_history.append({
                "role": "user",
                "content": user_question
            })
            
            # Get response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": answer
            })
            
            return answer
            
        except Exception as e:
            return f"Error getting AI response: {str(e)}"
    
    def _format_shap_values(self, shap_data: dict) -> str:
        """Format SHAP values for display."""
        if not shap_data or 'top_features' not in shap_data:
            return "No SHAP values available"
        
        top_features = shap_data.get('top_features', [])[:5]  # Top 5
        
        formatted = []
        for feature, importance in top_features:
            formatted.append(f"  • {feature}: {importance:.4f}")
        
        return "\n".join(formatted) if formatted else "No feature importance data"
    
    def reset_conversation(self):
        """Clear conversation history for new context."""
        self.conversation_history = []
    
    def get_suggested_questions(self, result_type: str) -> list:
        """
        Get suggested questions based on result type.
        
        Parameters
        ----------
        result_type : str
            'flow', 'finding', 'decision', or 'general'
        
        Returns
        -------
        list of str
            Suggested questions
        """
        suggestions = {
            'flow': [
                "Why was this classified as an attack?",
                "What features indicate this is malicious?",
                "What do the SHAP values mean?",
                "How confident is the model?",
                "Could this be a false positive?"
            ],
            'finding': [
                "Why is this finding important?",
                "What attack pattern was detected?",
                "Should I be concerned about this?",
                "What should I do next?",
                "How does this rule work?"
            ],
            'decision': [
                "Why did the AI choose this action?",
                "Was this the right decision?",
                "What alternatives were considered?",
                "Should this be escalated?",
                "How effective is this response?"
            ],
            'general': [
                "How does the threat hunting work?",
                "What does the detection rate mean?",
                "How accurate is the model?",
                "What are the different attack types?",
                "How does SHAP explainability help?"
            ]
        }
        
        return suggestions.get(result_type, suggestions['general'])
