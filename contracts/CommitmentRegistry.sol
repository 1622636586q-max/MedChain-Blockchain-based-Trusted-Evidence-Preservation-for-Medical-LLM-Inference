// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract CommitmentRegistry {
    struct CommitmentRecord {
        bytes32 commitmentHash;
        bytes32 modelHash;
        bytes32 requestIdHash;
        uint256 createdAt;
        address submitter;
    }

    mapping(bytes32 => CommitmentRecord) private records;
    mapping(bytes32 => bool) public exists;

    event CommitmentRecorded(
        bytes32 indexed requestIdHash,
        bytes32 indexed commitmentHash,
        bytes32 indexed modelHash,
        address submitter,
        uint256 createdAt
    );

    function recordCommitment(
        bytes32 requestIdHash,
        bytes32 commitmentHash,
        bytes32 modelHash
    ) external {
        require(requestIdHash != bytes32(0), "requestIdHash=0");
        require(commitmentHash != bytes32(0), "commitmentHash=0");
        require(!exists[requestIdHash], "request already recorded");

        records[requestIdHash] = CommitmentRecord({
            commitmentHash: commitmentHash,
            modelHash: modelHash,
            requestIdHash: requestIdHash,
            createdAt: block.timestamp,
            submitter: msg.sender
        });
        exists[requestIdHash] = true;

        emit CommitmentRecorded(
            requestIdHash,
            commitmentHash,
            modelHash,
            msg.sender,
            block.timestamp
        );
    }

    function getCommitment(
        bytes32 requestIdHash
    )
        external
        view
        returns (
            bool found,
            bytes32 commitmentHash,
            bytes32 modelHash,
            uint256 createdAt,
            address submitter
        )
    {
        if (!exists[requestIdHash]) {
            return (false, bytes32(0), bytes32(0), 0, address(0));
        }
        CommitmentRecord memory r = records[requestIdHash];
        return (true, r.commitmentHash, r.modelHash, r.createdAt, r.submitter);
    }
}
