# import os
# from typing import ClassVar

# import yaml
# from pydantic import BaseModel, Field, SecretStr, model_validator
# from pydantic_settings import BaseSettings, SettingsConfigDict

# # -----------------------------
# # Supabase settings
# # -----------------------------
# class FirstInfraSettings(BaseModel):
#     url: str = Field(default="", description="Supabase project URL")
#     key: SecretStr = Field(default=SecretStr(""), description="Supabase project API key")
#     table_name: str = Field(default="substack_articles", description="Supabase table name")


# # -----------------------------
# # Supabase database settings
# # -----------------------------
# class SecondInfraSettings(BaseModel):
#     host: str = Field(default="localhost", description="Database host")
#     name: str = Field(default="postgres", description="Database name")
#     user: str = Field(default="postgres", description="Database user")
#     password: SecretStr = Field(default=SecretStr("password"), description="Database password")
#     port: int = Field(default=6543, description="Database port")
#     test_database: str = Field(default="substack_test", description="Test database name")

# # # -----------------------------
# # # YAML loader
# # # -----------------------------
# # def load_yaml_feeds(path: str) -> list[FeedItem]:
# #     """
# #     Load RSS feed items from a YAML file.
# #     If the file does not exist or is empty, returns an empty list.

# #     Args:
# #         path (str): Path to the YAML file.

# #     Returns:
# #         list[FeedItem]: List of FeedItem instances loaded from the file.
# #     """
# #     if not os.path.exists(path):
# #         return []
# #     with open(path, encoding="utf-8") as f:
# #         data = yaml.safe_load(f)
# #     feed_list = data.get("feeds", [])
# #     return [FeedItem(**feed) for feed in feed_list]


# # -----------------------------
# # Main Settings
# # -----------------------------
# class Settings(BaseSettings):
#     firstinfra: FirstInfraSettings = Field(
#         default_factory=FirstInfraSettings
#     )
#     secondinfra: SecondInfraSettings = Field(
#         default_factory=SecondInfraSettings
#     )

#     # config_yaml_path: str = "src/configs/feeds.yaml"

#     # Pydantic v2 model config
#     model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
#         env_file=[".env"],
#         env_file_encoding="utf-8",
#         extra="ignore",
#         env_nested_delimiter="__",
#         case_sensitive=False,
#         frozen=True,
#     )

#     # @model_validator(mode="after")
#     # def load_yaml_feeds(self) -> "Settings":
#     #     """
#     #     Load RSS feeds from a YAML file after model initialization.
#     #     If the file does not exist or is empty, the feeds list remains unchanged.

#     #     Args:
#     #         self (Settings): The settings instance.

#     #     Returns:
#     #         Settings: The updated settings instance.
#     #     """
#     #     yaml_feeds = load_yaml_feeds(self.config_yaml_path)
#     #     if yaml_feeds:
#     #         self.rss.feeds = yaml_feeds
#     #     return self


# # -----------------------------
# # Instantiate settings
# # -----------------------------
# settings = Settings()