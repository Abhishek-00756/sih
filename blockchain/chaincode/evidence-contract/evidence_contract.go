package main

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

type EvidenceContract struct {
	contractapi.Contract
}

type Evidence struct {
	EventID        string `json:"event_id"`
	CameraID       string `json:"camera_id"`
	GlobalID       int    `json:"global_id"`
	EventType      string `json:"event_type"`
	Timestamp      string `json:"timestamp"`
	Direction      string `json:"direction,omitempty"`
	EvidenceSHA256 string `json:"evidence_sha256"`
	PreviousHash   string `json:"previous_hash,omitempty"`
}

func (c *EvidenceContract) RecordEvidence(
	ctx contractapi.TransactionContextInterface,
	eventID string,
	cameraID string,
	globalID int,
	eventType string,
	timestamp string,
	direction string,
	evidenceSHA256 string,
	previousHash string,
) error {

	if eventID == "" {
		return fmt.Errorf("eventID cannot be empty")
	}

	exists, err := c.EvidenceExists(ctx, eventID)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("evidence %s already exists", eventID)
	}

	if timestamp == "" {
		timestamp = time.Now().UTC().Format(time.RFC3339)
	}

	evidence := Evidence{
		EventID:        eventID,
		CameraID:       cameraID,
		GlobalID:       globalID,
		EventType:      eventType,
		Timestamp:      timestamp,
		Direction:      direction,
		EvidenceSHA256: evidenceSHA256,
		PreviousHash:   previousHash,
	}

	data, err := json.Marshal(evidence)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(eventID, data)
}

func (c *EvidenceContract) GetEvidence(
	ctx contractapi.TransactionContextInterface,
	eventID string,
) (*Evidence, error) {

	data, err := ctx.GetStub().GetState(eventID)
	if err != nil {
		return nil, fmt.Errorf("failed to read evidence %s: %v", eventID, err)
	}

	if data == nil {
		return nil, fmt.Errorf("evidence %s does not exist", eventID)
	}

	var evidence Evidence
	if err := json.Unmarshal(data, &evidence); err != nil {
		return nil, err
	}

	return &evidence, nil
}

func (c *EvidenceContract) EvidenceExists(
	ctx contractapi.TransactionContextInterface,
	eventID string,
) (bool, error) {

	data, err := ctx.GetStub().GetState(eventID)
	if err != nil {
		return false, err
	}

	return data != nil, nil
}

func (c *EvidenceContract) VerifyEvidence(
	ctx contractapi.TransactionContextInterface,
	eventID string,
	currentSHA256 string,
) (bool, error) {

	evidence, err := c.GetEvidence(ctx, eventID)
	if err != nil {
		return false, err
	}

	return evidence.EvidenceSHA256 == currentSHA256, nil
}

func (c *EvidenceContract) DeleteEvidence(
	ctx contractapi.TransactionContextInterface,
	eventID string,
) error {

	exists, err := c.EvidenceExists(ctx, eventID)
	if err != nil {
		return err
	}

	if !exists {
		return fmt.Errorf("evidence %s does not exist", eventID)
	}

	return ctx.GetStub().DelState(eventID)
}

func main() {
	chaincode, err := contractapi.NewChaincode(&EvidenceContract{})
	if err != nil {
		panic(err)
	}

	if err := chaincode.Start(); err != nil {
		panic(err)
	}
}
