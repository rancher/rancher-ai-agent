import os
import logging
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import CheckpointTuple

class MemoryManager:
    """
    Manages chat memories using a database or in-memory storage.
    """
    def __init__(self):
        self.db_enabled = os.environ.get("DB_ENABLED", "false").lower() == "true"
        self.db_url = os.environ.get("DB_CONNECTION_STRING", "")
        self.db_pool = None
        self.checkpointer = None
        
    async def initialize(self):
        if os.environ.get("DB_ENABLED", "false").lower() == "true":
            logging.info("Initializing PostgreSQL Checkpointer...")

            self.db_pool = AsyncConnectionPool(
                conninfo=self.db_url, 
                max_size=20,
                kwargs={"autocommit": True},
                open=False
            )
            
            await self.db_pool.open()

            self.checkpointer = AsyncPostgresSaver(self.db_pool)

            await self.checkpointer.setup()
            logging.info("PostgreSQL Checkpointer ready.")
        else:
            logging.info("Using InMemorySaver for checkpoints.")
            self.checkpointer = InMemorySaver()

    async def destroy(self):
        if self.db_pool:
            await self.db_pool.close()
            logging.info("PostgreSQL pool closed.")

    def get_checkpointer(self):
        if not self.checkpointer:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")
        return self.checkpointer

    def _filter_by_tags(self, tag_filters: list[dict], tags: list, role: str) -> bool:
        """
        TODO: add custom tag filtering logic here.
        Check if any of the tags are in the excluded tags list.
        
        Args:
            tag_filters: list of tag filter dicts
            tags: list of tags to check
            role: "human" or "ai"
        Returns:
            bool: True if none of the tags are in the excluded tags, False otherwise.
        """
        for tag in tags:
            for excluded in tag_filters:
                for tag_role, excluded_tags in excluded.items():
                    if tag_role == role and tag in excluded_tags:
                        return False
        return True
    
    def _is_empty_chat(self, checkpoint_tuple: CheckpointTuple, tag_filters: list[dict] = []) -> bool:
        """
        Check if a chat is empty from the checkpoint tuple.
        A chat is considered empty if it does not contain any messages besides tag-filtered messages.
        
        Example:
            tag_filters = [
                { "human": ["welcome"] },
                { "ai": ["welcome"] },
            ]
        """        
        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        agent_metatdata = channel_values.get("agent_metadata", {})
        messages = channel_values.get("messages", [])
        tags = agent_metatdata.get("tags", [])

        is_empty = True
        for msg in messages:
            if self._filter_by_tags(tag_filters, tags, msg.type):
                is_empty = False
                break
            
        return is_empty
    
    def _get_chat_metadata(self, checkpointTuple: CheckpointTuple) -> dict:
        """
        Extract chat metadata from a checkpoint tuple.
        """
        metadata = checkpointTuple.metadata
        channel_values = checkpointTuple.checkpoint.get("channel_values", {})

        name = metadata.get("chat_name", "")
        created_at = None
        
        messages = channel_values.get("messages", [])
        if messages and len(messages) > 0:
            # First message is used for chat metadata
            message = messages[0]

            additional_kwargs = message.additional_kwargs
            created_at = additional_kwargs.get("created_at") if additional_kwargs else None

            if not name:
                request_metadata = message.additional_kwargs.get("request_metadata", {})
                user_input = request_metadata.get("user_input", "")
                
                # Use content of first User message as default chat name
                name = user_input[:50] if user_input else "Untitled Chat"

        return {
            "name": name,
            "created_at": created_at
        }

    async def fetch_chats(self, user_id: str, filters: dict = {}) -> list:
        """
        TODO: implement filtering logic.
        Fetch chat threads from the database for a specific user.

        Args:
            user_id: The ID of the user.
            filters: A dictionary to filter chat threads.

        Returns:
            A list of chat thread records.
        """
        rows = []

        chat_ids = set()
        async for checkpoint_tuple in self.checkpointer.alist(config=None, filter={"user_id": user_id}):
            chat_id = checkpoint_tuple.config["configurable"]["thread_id"]
            user_id = checkpoint_tuple.metadata["user_id"]
            
            if not chat_id or not user_id:
                continue

            if self._is_empty_chat(checkpoint_tuple):
                logging.debug(f"Chat_id: {chat_id}, user_id: {user_id} is empty, skipping")
                continue
            
            logging.debug(f"Processing checkpoint_tuple for chat_id: {chat_id}, user_id: {user_id}")

            if chat_id not in chat_ids:
                chat_ids.add(chat_id)
                
                chat_metadata = self._get_chat_metadata(checkpoint_tuple)
                rows.append({
                    "id": chat_id,
                    "userId": user_id,
                    "name": chat_metadata.get("name"),
                    "createdAt": chat_metadata.get("created_at"),
                })

        return rows
    
    async def delete_chats(self, user_id: str) -> None:
        """
        Delete all chat threads for a specific user.

        Args:
            user_id: The ID of the user.
        """
        thread_ids = []
        async for checkpoint_tuple in self.checkpointer.alist(config=None, filter={"user_id": user_id}):
            thread_id = checkpoint_tuple.config['configurable']['thread_id']
            if thread_id not in thread_ids:
                thread_ids.append(thread_id)
        
        # Then delete them after iteration is complete
        for thread_id in thread_ids:
            await self.checkpointer.adelete_thread(thread_id)
            logging.debug(f"Deleted thread: {thread_id}, user_id: {user_id}")

    async def fetch_chat(self, chat_id: str, user_id: str) -> dict | None:
        """
        Fetch a specific chat thread from the database.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
        Returns:
            A chat thread record or None if not found.
        """
        config = {"configurable": {"thread_id": chat_id, "user_id": user_id}}
        checkpoint_tuple = await self.checkpointer.aget_tuple(config=config)

        if checkpoint_tuple:
            logging.debug(f"Found checkpoint_tuple for chat_id: {chat_id}, user_id: {user_id}")

            if self._is_empty_chat(checkpoint_tuple):
                logging.debug(f"Chat_id: {chat_id}, user_id: {user_id} is empty, skipping")
                return None

            chat_metadata = self._get_chat_metadata(checkpoint_tuple)
            chat = {
                "id": chat_id,
                "userId": user_id,
                "name": chat_metadata.get("name"),
                "createdAt": chat_metadata.get("created_at"),
            }
            return chat

        return None

    async def update_chat(self, chat_id: str, user_id: str, chat_data: dict) -> dict:
        """
        Update a specific chat thread for a specific user.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
            chat_data: The chat data to update.
        Returns:
            The updated chat thread record.
        """
        name = chat_data.get("name")
        
        if name and isinstance(name, str) and len(name) > 0:
            config = { "configurable": { "thread_id": chat_id, "user_id": user_id, "checkpoint_ns": "" }}
            
            checkpoint_tuple = await self.checkpointer.aget_tuple(config)
            
            if checkpoint_tuple:
                current_metadata = dict(checkpoint_tuple.metadata) if checkpoint_tuple.metadata else {}

                new_metadata = {**current_metadata, "chat_name": name}

                await self.checkpointer.aput(
                    config=config,
                    checkpoint=checkpoint_tuple.checkpoint,
                    metadata=new_metadata,
                    new_versions={}
                )

        chat = await self.fetch_chat(chat_id, user_id)

        return chat
    
    async def delete_chat(self, chat_id: str, user_id: str) -> None:
        """
        Delete a specific chat thread for a specific user.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
        """
        config = {"configurable": {"thread_id": chat_id}}
        checkpoint_tuple = await self.checkpointer.aget_tuple(config=config)

        if checkpoint_tuple and checkpoint_tuple.metadata.get("user_id") == user_id:
            await self.checkpointer.adelete_thread(chat_id)
            logging.debug(f"Deleted thread: {chat_id}, user_id: {user_id}")

    async def fetch_messages(self, chat_id: str, user_id: str, filters: dict = {}) -> list:
        """
        TODO: implement tags filtering logic.
        Fetch messages from the database based on filter configuration.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
            filters: A dictionary to filter messages.
        Returns:
            A list of message records.
            
        Filters example:
            tag_filters = [
                { "human": ["welcome"] },
                { "ai": ["welcome"] },
            ]
        """
        limit = filters.get("limit")

        rows = []
        
        config = {"configurable": {"thread_id": chat_id, "user_id": user_id}}
        checkpoint_tuple = await self.checkpointer.aget_tuple(config=config)

        all_messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", []) if checkpoint_tuple else []

        # Group messages by request_id
        messages_map = {}
        for message in all_messages:
            request_id = message.additional_kwargs["request_id"]
            if request_id:
                if request_id not in messages_map:
                    messages_map[request_id] = []
                messages_map[request_id].append(message)

        # Process messages for each request_id
        for request_id, messages in messages_map.items():
            logging.debug(f"Processing request_id: {request_id} with {len(messages)} messages")
            
            # Tags are propagated from user message to agent messages in the same request
            tags = []

            llm_str = ""
            mcp_str = ""

            for msg in messages:
                if msg.type == 'human':
                    request_metadata = msg.additional_kwargs.get("request_metadata", {})

                    tags = request_metadata.get("tags", [])
                    user_input = request_metadata.get("user_input", "")

                    if user_input:
                        rows.append({
                            "chatId": chat_id,
                            "role": "user",
                            "message": request_metadata.get("user_input", ""),
                            "context": request_metadata.get("context", None),
                            "tags": tags,
                            "createdAt": msg.additional_kwargs.get("created_at"),
                        })
                
                else:
                    if msg.type == 'ai':
                        llm_str = msg.content if msg.content else ""

                    if msg.type == 'tool':
                        interrupt_str = msg.additional_kwargs.get("interrupt_message", "")
                        confirmation = msg.additional_kwargs.get("confirmation", None)

                        if confirmation is not None and interrupt_str:
                            rows.append({
                                "chatId": chat_id,
                                "role": "agent",
                                "message": interrupt_str,
                                "confirmation": confirmation,
                                "createdAt": msg.additional_kwargs.get("created_at"),
                            })
                        else:
                            mcp_response = msg.additional_kwargs.get("mcp_response", "")
                            if mcp_response:
                                mcp_str = mcp_response
                                
            if mcp_str or llm_str:
                agent_response = mcp_str + llm_str
                if agent_response:
                    rows.append({
                        "chatId": chat_id,
                        "role": "agent",
                        "message": agent_response,
                        "tags": tags,
                        "createdAt": msg.additional_kwargs.get("created_at"),
                    })

            if limit and len(rows) >= limit:
                return rows[:limit]

        return rows

async def create_memory_manager() -> MemoryManager:
    """
    Factory function to create a MemoryManager instance.

    Returns:
        An instance of MemoryManager.
    """
    
    manager = MemoryManager()
    await manager.initialize()

    logging.info("MemoryManager created and initialized")
    return manager
