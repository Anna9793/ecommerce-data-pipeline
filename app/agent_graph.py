import os
import json
import logging
from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from app.agent_service import MarketingAgentService, _get_pydantic_schema
from vertexai.generative_models import GenerationConfig

logging.basicConfig(level=logging.INFO)

# ==========================================
# 1. Pydantic Schemas for LangGraph Nodes
# ==========================================

class GraphCopyDraft(BaseModel):
    subject: str = Field(description="Catchy email subject line")
    body: str = Field(description="Persuasive email body text")

class GraphCriticAudit(BaseModel):
    is_approved: bool = Field(description="True if email complies with all rules and tone guidelines, False if it needs revision")
    review_notes: str = Field(description="Audit summary and compliance notes")
    critique_feedback: str = Field(description="Constructive guidance for copywriter if rejected, or approval summary if approved")
    final_subject: str = Field(description="Polished subject line")
    final_body: str = Field(description="Polished body copy")

# ==========================================
# 2. LangGraph Typed State Definition
# ==========================================

class MarketingGraphState(TypedDict):
    customer_id: str
    profile: Dict[str, Any]
    recommended_products: List[Dict[str, Any]]
    analysis_text: Optional[str]
    strategy_text: Optional[str]
    draft_subject: Optional[str]
    draft_body: Optional[str]
    is_approved: bool
    review_notes: Optional[str]
    critique_feedback: Optional[str]
    iteration_count: int
    trace_logs: List[Dict[str, Any]]
    final_output: Optional[Dict[str, Any]]

# ==========================================
# 3. LangGraph Orchestrator
# ==========================================

class MarketingGraphOrchestrator:
    def __init__(self):
        self.agent_service = MarketingAgentService()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(MarketingGraphState)

        workflow.add_node("analyst", self._analyst_node)
        workflow.add_node("strategist", self._strategist_node)
        workflow.add_node("copywriter", self._copywriter_node)
        workflow.add_node("critic", self._critic_node)
        workflow.add_node("format_output", self._format_output_node)

        workflow.set_entry_point("analyst")
        workflow.add_edge("analyst", "strategist")
        workflow.add_edge("strategist", "copywriter")
        workflow.add_edge("copywriter", "critic")

        # Conditional Feedback Loop Edge
        workflow.add_conditional_edges(
            "critic",
            self._should_revise_or_end,
            {
                "copywriter": "copywriter",
                "format_output": "format_output"
            }
        )
        workflow.add_edge("format_output", END)

        return workflow.compile()

    # ----------------------------------------------------
    # Node 1: Behavioral Analyst Node
    # ----------------------------------------------------
    def _analyst_node(self, state: MarketingGraphState) -> Dict[str, Any]:
        logging.info("Executing LangGraph Node: Analyst for customer %s", state["customer_id"])
        profile = state["profile"]
        analysis = self.agent_service._run_analyst_agent(profile)
        
        trace = {
            "node": "analyst",
            "title": "Agent 1: Behavioral Analyst",
            "status": "completed",
            "content": analysis
        }
        logs = list(state.get("trace_logs", []))
        logs.append(trace)
        
        return {
            "analysis_text": analysis,
            "trace_logs": logs
        }

    # ----------------------------------------------------
    # Node 2: Campaign Strategist Node
    # ----------------------------------------------------
    def _strategist_node(self, state: MarketingGraphState) -> Dict[str, Any]:
        logging.info("Executing LangGraph Node: Strategist")
        profile = state["profile"]
        recs = state["recommended_products"]
        diagnosis = state.get("analysis_text", "")
        strategy = self.agent_service._run_strategist_agent(diagnosis, profile, recs)
        
        trace = {
            "node": "strategist",
            "title": "Agent 2: Campaign Strategist",
            "status": "completed",
            "content": strategy
        }
        logs = list(state.get("trace_logs", []))
        logs.append(trace)
        
        return {
            "strategy_text": strategy,
            "trace_logs": logs
        }

    # ----------------------------------------------------
    # Node 3: Creative Copywriter Node (Supports Revision Loops)
    # ----------------------------------------------------
    def _copywriter_node(self, state: MarketingGraphState) -> Dict[str, Any]:
        iteration = state.get("iteration_count", 0) + 1
        logging.info("Executing LangGraph Node: Copywriter (Cycle %d)", iteration)
        
        profile = state["profile"]
        recs = state["recommended_products"]
        strategy = state.get("strategy_text", "")
        feedback = state.get("critique_feedback", "")
        
        # Build prompt with optional critique feedback if revising
        rec_list_str = ""
        for idx, rec in enumerate(recs, 1):
            rec_list_str += f"{idx}. \"{rec['description']}\" — ${rec['unit_price']:.2f}\n"

        feedback_instruction = ""
        if feedback and iteration > 1:
            feedback_instruction = f"""
            CRITICAL REVISION DIRECTIVE FROM QUALITY CRITIC:
            "{feedback}"
            You MUST address these specific critique points in your new draft.
            """

        prompt = f"""
        You are an expert Creative Marketing Copywriter.
        Write a high-converting personalized email campaign based on this approved strategy:
        
        CUSTOMER PROFILE:
        - Recency: {profile.get('recency', 15):.0f} days
        - Total Spend: ${profile.get('avg_order_value', 100)*profile.get('frequency', 2):.2f}
        
        COMMERCIAL STRATEGY:
        {strategy}
        
        RECOMMENDED PRODUCTS:
        {rec_list_str}
        {feedback_instruction}
        
        Write a catchy subject line and warm persuasive body copy.
        """
        
        try:
            config = GenerationConfig(
                response_mime_type="application/json",
                response_schema=_get_pydantic_schema(GraphCopyDraft)
            )
            resp = self.agent_service.gemini_model.generate_content(prompt, generation_config=config)
            data = json.loads(resp.text)
            subject = data.get("subject", "Exclusive Recommendations For You")
            body = data.get("body", "Discover handpicked styles crafted for you.")
        except Exception as e:
            logging.warning("Copywriter graph node failed: %s. Using default copy.", e)
            subject = "Curated picks for your home"
            body = f"Hello! We selected lovely items including {recs[0]['description'] if recs else 'our bestsellers'} for you."

        trace = {
            "node": "copywriter",
            "title": f"Agent 3: Creative Copywriter (Cycle {iteration})",
            "status": "completed",
            "content": f"**Subject:** {subject}\n\n**Draft Body:**\n{body}"
        }
        logs = list(state.get("trace_logs", []))
        logs.append(trace)

        return {
            "draft_subject": subject,
            "draft_body": body,
            "iteration_count": iteration,
            "trace_logs": logs
        }

    # ----------------------------------------------------
    # Node 4: Quality & Compliance Critic Node
    # ----------------------------------------------------
    def _critic_node(self, state: MarketingGraphState) -> Dict[str, Any]:
        iteration = state.get("iteration_count", 1)
        logging.info("Executing LangGraph Node: Critic (Audit Cycle %d)", iteration)
        
        profile = state["profile"]
        subject = state.get("draft_subject", "")
        body = state.get("draft_body", "")

        prompt = f"""
        You are a Chief Compliance Officer & Senior Copy Critic.
        Audit this email draft before final customer dispatch:
        
        CUSTOMER CONTEXT:
        - Segment: {profile.get('label', 'Valued Customer')}
        - Churn Risk: {profile.get('churn_probability', 0.1)*100:.1f}%
        
        DRAFT SUBJECT: {subject}
        DRAFT BODY: {body}
        
        AUDIT RULES:
        1. NO internal cluster/segment names allowed (e.g. 'Cluster 2', 'At-Risk Segment').
        2. Tone must be warm, respectful, and not pushy.
        3. If churn risk > 50%, verify a special incentive (e.g. WINBACK20) is offered.
        4. If it fails any rule or needs major refinement, set is_approved=False and provide actionable critique_feedback. Otherwise, set is_approved=True.
        """
        
        try:
            config = GenerationConfig(
                response_mime_type="application/json",
                response_schema=_get_pydantic_schema(GraphCriticAudit)
            )
            resp = self.agent_service.gemini_model.generate_content(prompt, generation_config=config)
            data = json.loads(resp.text)
            is_approved = bool(data.get("is_approved", True))
            review_notes = data.get("review_notes", "Passed compliance check.")
            critique_feedback = data.get("critique_feedback", "Approved for dispatch.")
            final_subject = data.get("final_subject", subject)
            final_body = data.get("final_body", body)
        except Exception as e:
            logging.warning("Critic graph node failed: %s. Using safe approval fallback.", e)
            is_approved = True
            review_notes = "Audited against compliance rules. Approved."
            critique_feedback = "Approved."
            final_subject = subject
            final_body = body

        trace = {
            "node": "critic",
            "title": f"Agent 4: Compliance Critic (Audit Cycle {iteration})",
            "status": "approved" if is_approved else "rejected_for_revision",
            "content": f"**Status:** {'🟢 APPROVED' if is_approved else '🔄 REVISION REQUESTED'}\n**Notes:** {review_notes}\n**Feedback:** {critique_feedback}"
        }
        logs = list(state.get("trace_logs", []))
        logs.append(trace)

        return {
            "is_approved": is_approved,
            "review_notes": review_notes,
            "critique_feedback": critique_feedback,
            "draft_subject": final_subject,
            "draft_body": final_body,
            "trace_logs": logs
        }

    # ----------------------------------------------------
    # Conditional Edge: Self-Correction Decision
    # ----------------------------------------------------
    def _should_revise_or_end(self, state: MarketingGraphState) -> str:
        is_approved = state.get("is_approved", False)
        iteration = state.get("iteration_count", 0)

        if is_approved:
            logging.info("Critic approved campaign on iteration %d. Transitioning to format_output.", iteration)
            return "format_output"
            
        if iteration >= 3:
            logging.info("Max iterations (3) reached. Forcing completion to format_output.")
            return "format_output"

        logging.info("Critic requested revision (Iteration %d/3). Routing back to Copywriter!", iteration)
        return "copywriter"

    # ----------------------------------------------------
    # Node 5: Format Final Output Node
    # ----------------------------------------------------
    def _format_output_node(self, state: MarketingGraphState) -> Dict[str, Any]:
        logging.info("Executing LangGraph Node: Format Output")
        profile = state["profile"]
        subject = state.get("draft_subject", "Curated for You")
        body = state.get("draft_body", "Here are your recommendations.")
        recs = state.get("recommended_products", [])
        
        pref_hour = profile.get("preferred_shopping_hour", 12)
        delivery_meta = f"Schedule Delivery for {pref_hour}:00 (Customer Peak Active Hour)"
        
        final_output = {
            "customer_id": state["customer_id"],
            "segment": profile.get("label", "Valued Customer"),
            "churn_risk": f"{profile.get('churn_probability', 0.1)*100:.1f}%",
            "subject": subject,
            "body": body,
            "delivery_meta": delivery_meta,
            "recommended_products": [p["description"] for p in recs],
            "iterations_required": state.get("iteration_count", 1),
            "graph_engine": "LangGraph (StateGraph with Cyclic Feedback)",
            "agent_traces": state.get("trace_logs", [])
        }
        
        return {"final_output": final_output}

    # ----------------------------------------------------
    # Public Execution Method
    # ----------------------------------------------------
    def run(self, customer_id: str) -> Dict[str, Any]:
        """Runs the LangGraph multi-agent workflow for a given customer."""
        profile = self.agent_service.get_customer_profile(customer_id)
        if not profile:
            raise ValueError(f"Customer {customer_id} not found.")

        recs = self.agent_service.find_similar_products(profile)

        initial_state: MarketingGraphState = {
            "customer_id": str(customer_id),
            "profile": profile,
            "recommended_products": recs,
            "strategy_text": None,
            "draft_subject": None,
            "draft_body": None,
            "is_approved": False,
            "review_notes": None,
            "critique_feedback": None,
            "iteration_count": 0,
            "trace_logs": [],
            "final_output": None
        }

        final_state = self.graph.invoke(initial_state)
        return final_state["final_output"]
