# PostgreSQL Docker Setup

This project sets up a PostgreSQL database using Docker Compose. It includes the necessary configuration files to initialize and run the database in a containerized environment.

## Project Structure

```
postgres-docker
├── docker-compose.yml
├── init-db
│   └── init.sql
└── README.md
```

## Getting Started

To set up and run the PostgreSQL database, follow these steps:

1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd postgres-docker
   ```

2. **Build and Start the Containers**:
   Use Docker Compose to build and start the PostgreSQL service.
   ```bash
   docker-compose up -d
   ```

3. **Initialize the Database**:
   The `init.sql` file located in the `init-db` directory will be executed automatically to set up the database schema and initial data.

4. **Connect to the Database**:
   You can connect to the PostgreSQL database using any PostgreSQL client. The default connection parameters are:
   - **Host**: localhost
   - **Port**: 5432
   - **User**: postgres
   - **Password**: postgres
   - **Database**: postgres

5. **Stopping the Containers**:
   To stop the running containers, use:
   ```bash
   docker-compose down
   ```

## Additional Information

- Ensure that Docker and Docker Compose are installed on your machine.
- Modify the `docker-compose.yml` file to change database configurations as needed.
- Check the logs of the PostgreSQL container for any issues:
  ```bash
  docker-compose logs postgres
  ```

This setup provides a simple way to run a PostgreSQL database for development and testing purposes.